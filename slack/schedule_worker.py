import re
import sys
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from crawler.news.plan_sources_from_request import execute_rss_request
from crawler.rss_base_crawler import crawl_multiple_rss_sources
from slack.llm_router import (
    extract_json,
    format_seconds_hms,
    invoke_llm_with_fallback,
    to_nonneg_float,
)
from storage_backend import DEFAULT_SQLITE_PATH, get_active_storage_backend

_CRAWL_LOCK = threading.Lock()
_CRAWL_STATE_LOCK = threading.Lock()
_IS_CRAWLING = False

_SCHEDULE_LOCK = threading.Lock()
_SCHEDULE_STOP_EVENT: Optional[threading.Event] = None
_SCHEDULE_THREAD: Optional[threading.Thread] = None
_SCHEDULE_INFO: Dict[str, Any] = {"active": False}

_WEEKDAY_MAP = {
    "월": 0, "월요일": 0,
    "화": 1, "화요일": 1,
    "수": 2, "수요일": 2,
    "목": 3, "목요일": 3,
    "금": 4, "금요일": 4,
    "토": 5, "토요일": 5,
    "일": 6, "일요일": 6,
}


def is_crawling() -> bool:
    with _CRAWL_STATE_LOCK:
        return _IS_CRAWLING


def set_crawling(active: bool) -> None:
    global _IS_CRAWLING
    with _CRAWL_STATE_LOCK:
        _IS_CRAWLING = active


def _parse_hms_chunk(text: str) -> tuple[float, float, float]:
    h = sum(float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*시간", text))
    m = sum(float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*분", text))
    s = sum(float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*초", text))
    return h, m, s


def _resolve_target_datetime(user_text: str, now: datetime) -> Optional[datetime]:
    if "까지" not in user_text:
        return None

    until_part = user_text.split("까지", 1)[0]
    target_date = now.date()
    date_explicit = False

    m_ymd = re.search(r"(?:(\d{4})\s*년\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일", until_part)
    if m_ymd:
        year = int(m_ymd.group(1)) if m_ymd.group(1) else now.year
        month = int(m_ymd.group(2))
        day = int(m_ymd.group(3))
        target_date = datetime(year, month, day).date()
        if not m_ymd.group(1) and target_date < now.date():
            target_date = datetime(year + 1, month, day).date()
        date_explicit = True

    if not date_explicit:
        if "글피" in until_part:
            target_date = now.date() + timedelta(days=3)
            date_explicit = True
        elif "모레" in until_part:
            target_date = now.date() + timedelta(days=2)
            date_explicit = True
        elif "내일" in until_part:
            target_date = now.date() + timedelta(days=1)
            date_explicit = True
        else:
            m_days = re.search(r"(\d+)\s*일\s*(?:뒤|후)", until_part)
            m_weeks = re.search(r"(\d+)\s*주\s*(?:뒤|후)", until_part)
            if m_days or m_weeks:
                add_days = int(m_days.group(1)) if m_days else 0
                add_weeks = int(m_weeks.group(1)) if m_weeks else 0
                target_date = now.date() + timedelta(days=add_days, weeks=add_weeks)
                date_explicit = True

    if not date_explicit:
        m_wd = re.search(
            r"(?:(이번\s*주|다음\s*주|다다음\s*주)\s*)?(월요일|화요일|수요일|목요일|금요일|토요일|일요일|[월화수목금토일]요일)",
            until_part,
        )
        if m_wd:
            week_prefix = (m_wd.group(1) or "").replace(" ", "")
            wd_num = _WEEKDAY_MAP[m_wd.group(2)]
            days_ahead = (wd_num - now.weekday()) % 7
            if week_prefix == "다음주":
                days_ahead += 7
            elif week_prefix == "다다음주":
                days_ahead += 14
            target_date = now.date() + timedelta(days=days_ahead)
            date_explicit = True

    m_time = re.search(
        r"(오전|오후|밤|저녁|새벽|낮)?\s*(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?(?:\s*(\d{1,2})\s*초)?",
        until_part,
    )
    if "자정" in until_part and not m_time:
        hour, minute, second = 0, 0, 0
        if not date_explicit:
            target_date = now.date() + timedelta(days=1)
            date_explicit = True
    elif m_time:
        meridiem = m_time.group(1) or ""
        hour = int(m_time.group(2))
        minute = int(m_time.group(3) or 0)
        second = int(m_time.group(4) or 0)

        if meridiem in ("오후", "밤", "저녁") and 1 <= hour < 12:
            hour += 12
        elif meridiem == "낮" and 1 <= hour <= 5:
            hour += 12
        elif meridiem in ("오전", "새벽") and hour == 12:
            hour = 0
    else:
        if not date_explicit:
            return None
        hour, minute, second = 23, 59, 59

    candidate = datetime(
        target_date.year, target_date.month, target_date.day, hour, minute, second
    )
    if not date_explicit and candidate <= now:
        candidate += timedelta(days=1)

    return candidate


def _resolve_target_datetime_with_llm(
    user_text: str,
    now: datetime,
    chosen: Optional[Dict[str, Any]] = None,
) -> Optional[datetime]:
    if not (user_text or "").strip():
        return None

    chosen_hint = ""
    if isinstance(chosen, dict) and chosen.get("until_iso"):
        chosen_hint = f"\n초기 추출 힌트(until_iso): {chosen.get('until_iso')}"

    prompt = (
        "당신은 사용자 요청에서 종료/목표 시각('~까지')을 계산하는 시간 파서입니다.\n"
        "현재 기준 시각을 참고해 목표 시각을 'YYYY-MM-DDTHH:MM:SS' 형식 문자열로 반환하세요.\n"
        "규칙:\n"
        "1. 명확한 종료 시각/날짜/요일/상대시각('~까지', '내일 18시까지', '금요일 밤까지' 등)이 없으면 value를 null로 두세요.\n"
        "2. '반'은 30분, '정오'는 12:00:00, '자정'은 00:00:00(시각이 이미 지났거나 밤 자정 맥락이면 다음 날 00:00:00)으로 해석하세요.\n"
        "3. 마크다운, 코드블록, 설명 없이 오직 JSON 1개만 출력하세요.\n"
        '출력 형식: {"until_iso": "2026-11-27T18:00:00"} 또는 {"until_iso": null}\n\n'
        f"현재 시각: {now.isoformat(timespec='seconds')}\n"
        f"사용자 원문: {user_text!r}{chosen_hint}"
    )

    parsed = extract_json(
        invoke_llm_with_fallback(
            messages=[HumanMessage(content=prompt)],
            temperature=0.0,
            max_tokens=120,
            prefer_local=True,
        )
        or ""
    )

    if isinstance(parsed, dict) and "until_iso" in parsed:
        raw_iso = parsed.get("until_iso")
        if raw_iso is None or str(raw_iso).strip().lower() in ("", "null", "none"):
            return None
        try:
            return datetime.fromisoformat(str(raw_iso).strip())
        except Exception:
            pass

    return _resolve_target_datetime(user_text, now)


def _extract_request_text_from_user(user_text: str) -> str:
    cleaned = user_text.strip()
    if "까지" in cleaned:
        after_until = cleaned.split("까지", 1)[1].strip()
        if after_until:
            cleaned = after_until

    cleaned = re.sub(
        r"(오늘|내일|모레|글피|이번\s*주|다음\s*주|다다음\s*주)?\s*"
        r"(?:오전|오후|밤|저녁|새벽|낮)?\s*\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?(?:\s*\d{1,2}\s*초)?\s*까지",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"((?:\d+(?:\.\d+)?\s*(?:시간|분|초|일|주)\s*)+(?:동안|마다|텀|간격|씩|뒤|후))",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")
    return cleaned or user_text.strip()


def build_schedule_request_with_llm(
    user_text: str,
    chosen: Dict[str, Any],
    call_llm_fn: Optional[Any] = None,
) -> str:
    llm_hint = str(chosen.get("request_text") or "").strip()
    prompt = (
        "다음 사용자 요청에서 '기간(동안/까지)'과 '반복 간격(마다/텀/간격)' 표현만 제거하고, "
        "'수집 대상/분야/언론사/개수(N개씩, N건 등)/키워드'는 반드시 남긴 자연스러운 한국어 한 문장을 만드세요.\n"
        "예시:\n"
        "- 입력: '13시 20분까지 4분마다 30개씩 it 뉴스 가져와' -> 출력: '30개씩 it 뉴스 수집'\n"
        "- 입력: '2시간 동안 15분마다 경제 뉴스 3개씩 모아줘' -> 출력: '경제 뉴스 3개씩 수집'\n"
        "따옴표나 설명 없이 한 문장만 출력하세요.\n\n"
        f"사용자 원문: {user_text}\n"
        f"초기 추출 힌트: {llm_hint or '없음'}"
    )

    try:
        if callable(call_llm_fn):
            raw_out = call_llm_fn(prompt)
        else:
            raw_out = invoke_llm_with_fallback(
                messages=[
                    SystemMessage(content="오직 정제된 수집 요청 한 문장만 출력하세요."),
                    HumanMessage(content=prompt),
                ],
                temperature=0.0,
                max_tokens=150,
                prefer_local=True,
            )
        cleaned_llm = str(raw_out or "").strip().strip('"').strip("'").strip("`")
        if cleaned_llm:
            return cleaned_llm
    except Exception as e:
        print(f"[SCHEDULE_REQ_LLM] fallback due to error: {e}", file=sys.stderr)

    if llm_hint:
        return llm_hint
    return _extract_request_text_from_user(user_text)


def parse_schedule_plan(user_text: str, chosen: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now()

    int_h = to_nonneg_float(chosen.get("interval_hours"), 0.0)
    int_m = to_nonneg_float(chosen.get("interval_minutes"), 0.0)
    int_s = to_nonneg_float(chosen.get("interval_seconds"), 0.0)
    interval_sec = int_h * 3600.0 + int_m * 60.0 + int_s

    if interval_sec <= 0:
        m_every = re.search(r"((?:\d+(?:\.\d+)?\s*(?:시간|분|초)\s*)+(?:마다|텀|간격|씩))", user_text)
        if m_every:
            eh, em, es = _parse_hms_chunk(m_every.group(1))
            interval_sec = eh * 3600.0 + em * 60.0 + es

    if interval_sec <= 0:
        interval_sec = 30.0 * 60.0

    deadline: Optional[datetime] = None
    until_iso = str(chosen.get("until_iso") or "").strip()
    if until_iso and until_iso.lower() not in ("null", "none"):
        try:
            deadline = datetime.fromisoformat(until_iso)
        except Exception:
            deadline = None

    if deadline is None:
        deadline = _resolve_target_datetime_with_llm(user_text, now, chosen=chosen)

    if deadline is None:
        dur_h = to_nonneg_float(chosen.get("duration_hours"), 0.0)
        dur_m = to_nonneg_float(chosen.get("duration_minutes"), 0.0)
        dur_s = to_nonneg_float(chosen.get("duration_seconds"), 0.0)
        dur_sec = dur_h * 3600.0 + dur_m * 60.0 + dur_s

        if dur_sec <= 0:
            m_dur = re.search(r"((?:\d+(?:\.\d+)?\s*(?:시간|분|초)\s*)+)동안", user_text)
            if m_dur:
                dh, dm, ds = _parse_hms_chunk(m_dur.group(1))
                dur_sec = dh * 3600.0 + dm * 60.0 + ds

        if dur_sec <= 0:
            dur_sec = 3600.0

        deadline = now + timedelta(seconds=dur_sec)

    llm_req = str(chosen.get("request_text") or "").strip()
    request_text = (
        llm_req
        if llm_req and llm_req.lower() not in ("null", "none")
        else _extract_request_text_from_user(user_text)
    )

    return {
        "started_at": now,
        "deadline": deadline,
        "interval_sec": max(5.0, interval_sec),
        "request_text": request_text,
    }


def _build_multi_rss_configs(request_text: str, sqlite_path: str, mcp_tools: Any) -> List[Dict[str, Any]]:
    try:
        from crawler.news.plan_sources_from_request import plan_sources_from_request

        planned = plan_sources_from_request(request_text)
        if isinstance(planned, dict) and planned.get("configs"):
            configs = []
            for cfg in planned["configs"]:
                item = dict(cfg)
                item.setdefault("sqlite_path", sqlite_path)
                configs.append(item)
            return configs
    except Exception:
        pass

    if mcp_tools is not None and hasattr(mcp_tools, "build_rss_sources_config"):
        built = mcp_tools.build_rss_sources_config(request_text)
        if isinstance(built, list) and built:
            return built

    return []


def _post_schedule_msg(text: str, channel_id: Optional[str], client: Any, say: Any) -> None:
    if channel_id and client:
        try:
            client.chat_postMessage(channel=channel_id, text=text)
            return
        except Exception:
            pass
    try:
        say(text=text)
    except Exception:
        pass


def _run_single_scheduled_tick(
    request_text: str,
    channel_id: Optional[str],
    client: Any,
    say: Any,
    run_idx: int,
    deadline: datetime,
    next_run_at: Optional[datetime],
    mcp_tools: Any = None,
) -> None:
    next_str = next_run_at.strftime("%H:%M:%S") if next_run_at else "종료 예정"
    deadline_str = deadline.strftime("%H:%M:%S")

    _post_schedule_msg(
        text=(
            f"⏳ *[주기 수집 #{run_idx} 가동]* `{request_text}`\n"
            f"• 마감 시각: `{deadline_str}` | 다음 회차: `{next_str}`"
        ),
        channel_id=channel_id,
        client=client,
        say=say,
    )

    if not _CRAWL_LOCK.acquire(blocking=False):
        _post_schedule_msg(
            text=f"⏭️ *[주기 수집 #{run_idx} 건너뜀]* 다른 수집 작업이 진행 중이라 다음 회차(`{next_str}`)에 이어갑니다.",
            channel_id=channel_id,
            client=client,
            say=say,
        )
        return

    set_crawling(True)
    try:
        configs = _build_multi_rss_configs(request_text, DEFAULT_SQLITE_PATH, mcp_tools)
        if configs:
            saved_total = crawl_multiple_rss_sources(
                sources_config=configs,
                storage_mode=get_active_storage_backend(),
                sqlite_path=DEFAULT_SQLITE_PATH,
                on_progress=None,
            )
        else:
            data = execute_rss_request(
                request_text=request_text,
                sqlite_path=DEFAULT_SQLITE_PATH,
                on_progress=None,
            )
            saved_total = int(data.get("processed_total") or 0)

        rem_sec = max(0.0, (deadline - datetime.now()).total_seconds())
        _post_schedule_msg(
            text=(
                f"✅ *[주기 수집 #{run_idx} 완료]* 저장 `{saved_total}`건\n"
                f"• 남은 시간: `{format_seconds_hms(rem_sec)}` | 다음 회차: `{next_str}`"
            ),
            channel_id=channel_id,
            client=client,
            say=say,
        )
    except Exception as e:
        print(f"[SCHED] tick #{run_idx} error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        _post_schedule_msg(
            text=f"⚠️ *[주기 수집 #{run_idx} 오류]* `{e}` (다음 회차: `{next_str}`)",
            channel_id=channel_id,
            client=client,
            say=say,
        )
    finally:
        set_crawling(False)
        _CRAWL_LOCK.release()


def run_once_rss_request(
    request_text: str,
    say: Any,
    mcp_tools: Any = None,
) -> Dict[str, Any]:
    if not _CRAWL_LOCK.acquire(blocking=False):
        say(text="현재 다른 뉴스 수집 작업이 진행 중이에요. 잠시 후 다시 시도해 주세요.")
        return {"busy": True}

    set_crawling(True)
    try:
        def _on_progress(msg: str) -> None:
            if msg and msg.strip():
                say(text=msg.strip())

        data = execute_rss_request(
            request_text=request_text,
            sqlite_path=DEFAULT_SQLITE_PATH,
            on_progress=_on_progress,
        )
    finally:
        set_crawling(False)
        _CRAWL_LOCK.release()

    if data.get("help"):
        say(text=data.get("guide", ""))
        return data

    planned = data.get("planned") or {}
    unsupported = planned.get("unsupported_items") or []
    if unsupported:
        say(text=f"⚠️ 일부 미지원 항목 제외: {', '.join(str(x) for x in unsupported)}")

    if data.get("skipped_crawl"):
        say(text=f"❌ {data.get('message') or '수집할 대상이 없어 건너뛰었어요.'}")
        return data

    sources = planned.get("configs") or []
    lines = [
        f"• {s['source_name']} ({s.get('category', '종합')}): {s.get('limit')}개"
        for s in sources
    ]
    say(
        text=(
            f"✅ 수집 완료 (처리 `{data.get('processed_total', 0)}`건)\n"
            + ("\n".join(lines) if lines else "• 매칭된 소스가 없어요.")
        )
    )
    return data


def start_rss_schedule(
    plan: Dict[str, Any],
    channel_id: Optional[str],
    client: Any,
    say: Any,
    mcp_tools: Any = None,
) -> Dict[str, Any]:
    global _SCHEDULE_STOP_EVENT, _SCHEDULE_THREAD, _SCHEDULE_INFO

    started_at: datetime = plan["started_at"]
    deadline: datetime = plan["deadline"]
    interval_sec: float = plan["interval_sec"]
    request_text: str = plan["request_text"]

    if deadline <= datetime.now():
        raise ValueError("종료 시각이 현재 시각보다 이전입니다.")

    with _SCHEDULE_LOCK:
        if _SCHEDULE_STOP_EVENT is not None:
            _SCHEDULE_STOP_EVENT.set()
        stop_event = threading.Event()
        _SCHEDULE_STOP_EVENT = stop_event
        _SCHEDULE_INFO = {
            "active": True,
            "started_at": started_at.isoformat(timespec="seconds"),
            "deadline": deadline.isoformat(timespec="seconds"),
            "interval_sec": interval_sec,
            "request_text": request_text,
        }

        def _worker() -> None:
            global _SCHEDULE_STOP_EVENT, _SCHEDULE_INFO
            run_idx = 0
            next_run_at = datetime.now()

            while not stop_event.is_set():
                if datetime.now() > deadline:
                    break

                run_idx += 1
                candidate_next = next_run_at + timedelta(seconds=interval_sec)
                while candidate_next <= datetime.now():
                    candidate_next += timedelta(seconds=interval_sec)
                upcoming_next = candidate_next if candidate_next <= deadline else None

                _run_single_scheduled_tick(
                    request_text=request_text,
                    channel_id=channel_id,
                    client=client,
                    say=say,
                    run_idx=run_idx,
                    deadline=deadline,
                    next_run_at=upcoming_next,
                    mcp_tools=mcp_tools,
                )

                next_run_at = candidate_next
                if next_run_at > deadline:
                    break

                wait_sec = (next_run_at - datetime.now()).total_seconds()
                if wait_sec > 0 and stop_event.wait(timeout=wait_sec):
                    break

            with _SCHEDULE_LOCK:
                if _SCHEDULE_STOP_EVENT is stop_event:
                    _SCHEDULE_INFO = {"active": False}
                    _SCHEDULE_STOP_EVENT = None

            if not stop_event.is_set():
                _post_schedule_msg(
                    text=(
                        f"🏁 *주기 RSS 수집 종료* (`{deadline.strftime('%H:%M:%S')}` 도달, 총 `{run_idx}`회 실행)\n"
                        f"• 수집 요청: `{request_text}`"
                    ),
                    channel_id=channel_id,
                    client=client,
                    say=say,
                )

        t = threading.Thread(target=_worker, name="rss-periodic-collector", daemon=True)
        _SCHEDULE_THREAD = t
        t.start()

    return plan


def stop_rss_schedule() -> Dict[str, Any]:
    global _SCHEDULE_STOP_EVENT, _SCHEDULE_INFO

    with _SCHEDULE_LOCK:
        was_active = bool(_SCHEDULE_INFO.get("active"))
        prev_info = dict(_SCHEDULE_INFO)
        if _SCHEDULE_STOP_EVENT is not None:
            _SCHEDULE_STOP_EVENT.set()
            _SCHEDULE_STOP_EVENT = None
        _SCHEDULE_INFO = {"active": False}

    return {
        "stopped": was_active,
        "prev_info": prev_info,
        "is_crawling_now": is_crawling(),
    }


def handle_stop_command(say: Any) -> None:
    res = stop_rss_schedule()
    if res["stopped"]:
        extra = (
            "\n• (현재 실행 중이던 1회 수집만 마무리된 뒤 다음 회차부터 즉시 중단됩니다.)"
            if res["is_crawling_now"]
            else ""
        )
        say(text=f"🛑 *주기 RSS 수집을 즉시 중단했어요.*{extra}")
    elif res["is_crawling_now"]:
        say(text="🛑 예약된 주기 수집은 없지만, 현재 진행 중인 단발 수집이 끝나면 더 이상 실행되지 않아요.")
    else:
        say(text="ℹ️ 현재 실행 중인 주기 수집 작업이 없어요.")