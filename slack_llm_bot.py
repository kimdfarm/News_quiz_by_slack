import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MCP_DIR = os.path.join(_BASE_DIR, "mcp")

for _p in (_BASE_DIR, _MCP_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from crawler.news.plan_sources_from_request import (
    execute_rss_request,
    get_rss_crawling_guide,
    parse_help_topic,
)
from crawler.rss_base_crawler import crawl_multiple_rss_sources
from sqlite_store import init_sqlite_db
from storage_backend import (
    DEFAULT_SQLITE_PATH,
    get_active_storage_backend,
    import_sqlite_file_data,
)
from tool_rag_router import MCP_TOOL_REGISTRY, retrieve_top_tools

load_dotenv()


def _resolve_server_path() -> str:
    candidates = [
        os.path.join(_MCP_DIR, "news_mcp_server.py"),
        os.path.join(_BASE_DIR, "news_mcp_server.py"),
        os.path.join(os.path.dirname(_BASE_DIR), "mcp", "news_mcp_server.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


_SERVER_PATH = _resolve_server_path()
_spec = importlib.util.spec_from_file_location("news_mcp_server", _SERVER_PATH)
mcp_tools = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(mcp_tools)

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

VALID_TOOLS = {item["tool"] for item in MCP_TOOL_REGISTRY} | {"none"}

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


# ---------------------------------------------------------------------------
# 상태 및 공통 파싱 유틸리티
# ---------------------------------------------------------------------------
def is_crawling() -> bool:
    with _CRAWL_STATE_LOCK:
        return _IS_CRAWLING


def set_crawling(active: bool) -> None:
    global _IS_CRAWLING
    with _CRAWL_STATE_LOCK:
        _IS_CRAWLING = active


def _clean_mention_text(text: str) -> str:
    return re.sub(r"<@[A-Z0-9_]+>", "", text or "").strip()


def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None


def _normalize_backend_value(raw_val: Any) -> Optional[str]:
    if raw_val is None:
        return None
    val = str(raw_val).strip().lower()
    if not val or val in ("null", "none", "auto", "default", "미지정"):
        return None
    if val in ("sqlite", "file", "local", "로컬", "파일"):
        return "sqlite"
    if val in ("postgres", "postgresql", "neon", "네온", "네온db", "db", "디비", "디지털 db"):
        return "postgres"
    return None


def _normalize_positive_int(raw_val: Any, default: Optional[int] = None) -> Optional[int]:
    if raw_val is None or isinstance(raw_val, bool):
        return default
    if isinstance(raw_val, int):
        return raw_val if raw_val > 0 else None
    if isinstance(raw_val, str):
        cleaned = raw_val.strip()
        if not cleaned or cleaned.lower() in ("null", "none", "all", "전체", "모두", "다"):
            return None
        if cleaned.isdigit():
            num = int(cleaned)
            return num if num > 0 else None
    return default


def _to_nonneg_float(val: Any, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        num = float(str(val).strip())
        return num if num >= 0 else default
    except Exception:
        return default


def _format_seconds_hms(total_sec: float) -> str:
    sec_int = max(0, int(round(total_sec)))
    h, rem = divmod(sec_int, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}시간")
    if m:
        parts.append(f"{m}분")
    if s or not parts:
        parts.append(f"{s}초")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# LLM 호출 (Groq + Ollama 폴백) 및 도구 라우팅
# ---------------------------------------------------------------------------
def _is_ollama_model_available(model_name: str, base_url: str, timeout: float = 1.5) -> bool:
    target = (model_name or "").strip().lower()
    if not target:
        return False
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return False
        payload = resp.json() if resp.content else {}
        for item in payload.get("models") or []:
            name = str((item or {}).get("name") or (item or {}).get("model") or "").strip().lower()
            if name == target or name.startswith(f"{target}:"):
                return True
        return False
    except Exception:
        return False


def _invoke_llm_with_fallback(
    messages: List[BaseMessage],
    temperature: float = 0.0,
    max_tokens: int = 600,
    prefer_local: bool = False,
) -> Optional[str]:
    groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    groq_models = list(
        dict.fromkeys(
            m.strip()
            for m in [
                (os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b").strip(),
                "llama-3.3-70b-versatile",
                "openai/gpt-oss-20b",
            ]
            if m and m.strip()
        )
    )

    ollama_model = (os.getenv("OLLAMA_MODEL") or "gemma3:4b").strip()
    ollama_base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()

    def _run_groq() -> Optional[str]:
        if not groq_api_key:
            return None
        for model_name in groq_models:
            try:
                llm = ChatGroq(
                    model=model_name,
                    api_key=groq_api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                resp = llm.invoke(messages)
                return resp.content if isinstance(resp.content, str) else str(resp.content)
            except Exception as e:
                print(f"[!] Groq({model_name}) 호출 실패: {e}", file=sys.stderr)
        return None

    def _run_ollama() -> Optional[str]:
        if not _is_ollama_model_available(ollama_model, ollama_base_url):
            print(
                f"[i] 로컬 Ollama에 '{ollama_model}' 모델이 없어 가동하지 않습니다.",
                file=sys.stderr,
            )
            return None
        try:
            llm = ChatOllama(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=temperature,
            )
            resp = llm.invoke(messages)
            return resp.content if isinstance(resp.content, str) else str(resp.content)
        except Exception as e:
            print(f"[!] Ollama({ollama_model}) 호출 실패: {e}", file=sys.stderr)
            return None

    if prefer_local:
        local_res = _run_ollama()
        return local_res if local_res is not None else _run_groq()

    groq_res = _run_groq()
    if groq_res is not None:
        return groq_res
    return _run_ollama()


def _choose_mcp_tool(user_text: str) -> Dict[str, Any]:
    """키워드/태그/그룹(+선택적 임베딩)으로 추린 후보 도구만 LLM에 넘겨 JSON 인자를 추출."""
    top_tools = retrieve_top_tools(user_text, top_k=3, min_score=1.5)
    if not top_tools:
        print(f"[DEBUG] text={user_text!r} -> no matching tools (score below threshold)", file=sys.stderr)
        return {"tool": "none"}

    compact_candidates = [
        {
            "tool": t["tool"],
            "description": t["description"],
            "examples": (t.get("examples") or [])[:3],
            "schema": t["schema"],
        }
        for t in top_tools
    ]

    system_prompt = (
        "당신은 MCP 도구 호출 JSON 생성기입니다. "
        "설명이나 마크다운 없이 오직 JSON 객체 1개만 출력하세요.\n"
        f"후보 도구 목록:\n{json.dumps(compact_candidates, ensure_ascii=False)}\n\n"
        "규칙:\n"
        "- 후보 중 요청 의도에 맞는 도구를 고르고 해당 schema 형식대로 JSON을 채우세요.\n"
        "- 기간(~동안, ~까지, 내일/모레/요일/날짜)이나 반복 간격(~마다, ~텀, ~간격, ~씩)이 있으면 "
        "반드시 schedule_rss_collection을 선택하고, 1회성 수집 요청이면 run_rss_by_request를 선택하세요.\n"
        "- 어느 도구에도 해당하지 않는 일반 대화면 {\"tool\": \"none\"}을 출력하세요.\n"
        "- delete_news_records일 때 source에는 언론사/섹션명만 넣고('정보','기사','데이터' 제외), "
        "'오래된/옛날/가장 먼저'면 order='oldest', '최근/최신'이면 order='newest'로 하세요.\n"
        "- delete_news_records에서 source/category/keyword/days/limit 중 하나라도 있으면 delete_all은 반드시 false로 하세요."
    )

    for attempt in range(2):
        raw = _invoke_llm_with_fallback(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"요청: {user_text}\nJSON:"),
            ],
            temperature=0.0,
            max_tokens=400,
            prefer_local=False,
        )
        parsed = _extract_json(raw or "")
        print(
            f"[DEBUG] text={user_text!r} top={[t['tool'] for t in top_tools]} "
            f"attempt={attempt+1} -> parsed={parsed}",
            file=sys.stderr,
        )
        if parsed and parsed.get("tool") in VALID_TOOLS:
            return parsed

    return {"tool": "none"}


def ask_llm_for_quiz(prompt: str) -> str:
    prefer_local = (os.getenv("QUIZ_LLM_PROVIDER") or "").strip().lower() == "ollama"
    out = _invoke_llm_with_fallback(
        messages=[HumanMessage(content=prompt)],
        temperature=0.3,
        max_tokens=1200,
        prefer_local=prefer_local,
    )
    return out or "사용 가능한 LLM 모델이 없어 퀴즈를 생성하지 못했습니다."


# ---------------------------------------------------------------------------
# 주기 RSS 수집 스케줄링 및 멀티 RSS 실행
# ---------------------------------------------------------------------------
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
    """user_text에서 기간/주기 표현을 제외한 순수 RSS 수집 요청을 LLM으로 정제."""
    llm_hint = str(chosen.get("request_text") or "").strip()
    prompt = (
        "다음 사용자 요청에서 '기간(동안/까지)'과 '반복 간격(마다/텀/간격/씩)'에 해당하는 시간 표현만 제거하고, "
        "RSS 뉴스 수집에 필요한 '수집 대상/분야/언론사/개수/키워드' 요청만 자연스러운 한국어 한 문장으로 남겨줘.\n"
        "특정 대상이나 개수가 없다면 '요청된 조건에 맞춰 뉴스 수집'처럼 원문 의도를 살려 짧게 요약하고, "
        "따옴표나 부가 설명 없이 요청 문장만 출력해.\n\n"
        f"사용자 원문: {user_text}\n"
        f"초기 추출 힌트: {llm_hint or '없음'}"
    )

    try:
        if callable(call_llm_fn):
            raw_out = call_llm_fn(prompt)
        else:
            raw_out = _invoke_llm_with_fallback(
                messages=[
                    SystemMessage(content="오직 정제된 수집 요청 한 문장만 출력하세요."),
                    HumanMessage(content=prompt),
                ],
                temperature=0.0,
                max_tokens=150,
                prefer_local=False,
            )
        cleaned_llm = str(raw_out or "").strip().strip('"').strip("'").strip("`")
        if cleaned_llm:
            return cleaned_llm
    except Exception as e:
        print(f"[SCHEDULE_REQ_LLM] fallback due to error: {e}", file=sys.stderr)

    if llm_hint:
        return llm_hint
    return _extract_request_text_from_user(user_text)


def _parse_schedule_plan(user_text: str, chosen: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now()

    int_h = _to_nonneg_float(chosen.get("interval_hours"), 0.0)
    int_m = _to_nonneg_float(chosen.get("interval_minutes"), 0.0)
    int_s = _to_nonneg_float(chosen.get("interval_seconds"), 0.0)
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
        deadline = _resolve_target_datetime(user_text, now)

    if deadline is None:
        dur_h = _to_nonneg_float(chosen.get("duration_hours"), 0.0)
        dur_m = _to_nonneg_float(chosen.get("duration_minutes"), 0.0)
        dur_s = _to_nonneg_float(chosen.get("duration_seconds"), 0.0)
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


def _build_multi_rss_configs(request_text: str, sqlite_path: str) -> List[Dict[str, Any]]:
    try:
        from crawler.news.plan_sources_from_request import plan_sources_from_request  # type: ignore

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

    if hasattr(mcp_tools, "build_rss_sources_config"):
        built = mcp_tools.build_rss_sources_config(request_text)
        if isinstance(built, list) and built:
            return built

    return []


def _post_schedule_msg(
    text: str,
    channel_id: Optional[str],
    client: Any,
    say: Any,
) -> None:
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
) -> None:
    print(f"[SCHED] tick #{run_idx} start: {request_text!r}", file=sys.stderr, flush=True)
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
        print(f"[SCHED] tick #{run_idx} skipped (crawl lock busy)", file=sys.stderr, flush=True)
        _post_schedule_msg(
            text=f"⏭️ *[주기 수집 #{run_idx} 건너뜀]* 다른 수집 작업이 진행 중이라 다음 회차(`{next_str}`)에 이어갑니다.",
            channel_id=channel_id,
            client=client,
            say=say,
        )
        return

    set_crawling(True)
    try:
        print(f"[SCHED] tick #{run_idx} before _build_multi_rss_configs", file=sys.stderr, flush=True)
        configs = _build_multi_rss_configs(request_text, DEFAULT_SQLITE_PATH)
        print(f"[SCHED] tick #{run_idx} after _build_multi_rss_configs: {len(configs)}", file=sys.stderr, flush=True)

        if configs:
            print(f"[SCHED] tick #{run_idx} calling crawl_multiple_rss_sources", file=sys.stderr, flush=True)
            saved_total = crawl_multiple_rss_sources(
                sources_config=configs,
                storage_mode=get_active_storage_backend(),
                sqlite_path=DEFAULT_SQLITE_PATH,
                on_progress=None,
            )
        else:
            print(f"[SCHED] tick #{run_idx} calling execute_rss_request", file=sys.stderr, flush=True)
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
                f"• 남은 시간: `{_format_seconds_hms(rem_sec)}` | 다음 회차: `{next_str}`"
            ),
            channel_id=channel_id,
            client=client,
            say=say,
        )
        print(f"[SCHED] tick #{run_idx} done (saved={saved_total})", file=sys.stderr, flush=True)
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
        print(f"[SCHED] tick #{run_idx} released lock", file=sys.stderr, flush=True)

def start_rss_schedule(
    plan: Dict[str, Any],
    channel_id: Optional[str],
    client: Any,
    say: Any,
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
            global _SCHEDULE_STOP_EVENT, _SCHEDULE_INFO  # <-- 이 줄 추가
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


def _handle_stop_command(say: Any) -> None:
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


# ---------------------------------------------------------------------------
# 파일 업로드/복원 및 출력 포맷 헬퍼
# ---------------------------------------------------------------------------
def _download_slack_file(file_info: Dict[str, Any]) -> str:
    url = file_info.get("url_private_download") or file_info.get("url_private")
    if not url:
        raise ValueError("Slack 파일 다운로드 URL이 없어요.")

    bot_token = (os.environ.get("SLACK_BOT_TOKEN") or "").strip()
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {bot_token}"},
        timeout=30,
    )
    resp.raise_for_status()

    fd, tmp_path = tempfile.mkstemp(prefix="slack_import_", suffix=".sqlite")
    try:
        with os.fdopen(fd, "wb") as tmp_f:
            tmp_f.write(resp.content)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return tmp_path


def _process_slack_files(attached_files: List[Dict[str, Any]], say: Any) -> None:
    if not attached_files:
        return

    with _IMPORT_LOCK:
        for file_info in attached_files:
            orig_name = file_info.get("name") or "uploaded_file"
            tmp_path = None
            try:
                tmp_path = _download_slack_file(file_info)
                result = import_sqlite_file_data(
                    uploaded_sqlite_path=tmp_path,
                    target_sqlite_path=DEFAULT_SQLITE_PATH,
                )
            except Exception as e:
                say(text=f"❌ `{orig_name}` 파일을 가져오는 중 실패했어요: {e}")
                continue
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            status = result.get("status")
            if status == "ok":
                say(
                    text=(
                        f"📥 *SQLite 데이터 복원/병합 완료* (`{orig_name}`)\n"
                        f"• 파일 내 전체 기사: {result.get('fetched_count', 0)}건\n"
                        f"• 신규 반영: *{result.get('inserted_count', 0)}건*\n"
                        f"• 중복 건너뜀: {result.get('skipped_count', 0)}건"
                    )
                )
            elif status == "rejected":
                say(
                    text=(
                        f"🚫 *파일 반영 거절* (`{orig_name}`)\n"
                        f"• 사유: {result.get('message')}\n"
                        f"• 기존 저장소 데이터는 변경되지 않았어요."
                    )
                )
            else:
                say(text=f"⚠️ 복원 실패 (`{orig_name}`): {result.get('message')}")


def _format_sync_result(data: dict, direction_label: str) -> str:
    if data.get("status") != "ok":
        return f"⚠️ 동기화 실패 ({direction_label}): {data.get('message', '알 수 없는 오류')}"

    limit_val = data.get("limit")
    limit_text = f"최근 {limit_val}건" if limit_val else "전체"
    return (
        f"🔄 *동기화 완료* ({direction_label})\n"
        f"• 대상 범위: {limit_text} (조회 {data.get('fetched_count', 0)}건)\n"
        f"• 신규 반영: *{data.get('inserted_count', 0)}건*\n"
        f"• 중복 건너뜀: {data.get('skipped_count', 0)}건"
    )


# ---------------------------------------------------------------------------
# Slack 이벤트 핸들러
# ---------------------------------------------------------------------------
@app.event("app_mention")
def handle_mention(event, say, client):
    user_text = _clean_mention_text(event.get("text", ""))

    if user_text.strip().lower() == "!stop":
        _handle_stop_command(say)
        return

    help_topic = parse_help_topic(user_text)
    if help_topic:
        say(text=get_rss_crawling_guide(help_topic))
        return

    attached_files = event.get("files") or []
    if attached_files:
        _process_slack_files(attached_files, say)
        if not user_text:
            return

    if not user_text:
        say(text="명령을 입력해 주세요. (`--help`로 확인)")
        return

    chosen = _choose_mcp_tool(user_text)
    tool = chosen.get("tool", "none")

    if tool == "none":
        say(text="그런 기능은 없어요.")
        return

    if tool == "set_storage_backend":
        target_mode = _normalize_backend_value(chosen.get("storage_backend"))
        if not target_mode:
            target_mode = "postgres" if get_active_storage_backend() == "sqlite" else "sqlite"
        raw_res = mcp_tools.set_storage_backend(target_mode)
        data = json.loads(raw_res)
        say(text=f"✅ 저장소 백엔드를 `{data.get('STORAGE_BACKEND')}`로 변경했어요.")
        return

    if tool == "run_rss_by_request":
        if not _CRAWL_LOCK.acquire(blocking=False):
            say(text="현재 다른 뉴스 수집 작업이 진행 중이에요. 잠시 후 다시 시도해 주세요.")
            return

        set_crawling(True)
        try:
            def _on_progress(msg: str) -> None:
                if msg and msg.strip():
                    say(text=msg.strip())

            data = execute_rss_request(
                request_text=user_text,
                sqlite_path=DEFAULT_SQLITE_PATH,
                on_progress=_on_progress,
            )
        finally:
            set_crawling(False)
            _CRAWL_LOCK.release()

        if data.get("help"):
            say(text=data.get("guide", ""))
            return

        planned = data.get("planned") or {}
        unsupported = planned.get("unsupported_items") or []
        if unsupported:
            say(text=f"⚠️ 일부 미지원 항목 제외: {', '.join(str(x) for x in unsupported)}")

        if data.get("skipped_crawl"):
            say(text=f"❌ {data.get('message') or '수집할 대상이 없어 건너뛰었어요.'}")
            return

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
        return

    if tool == "schedule_rss_collection":
        try:
            plan = _parse_schedule_plan(user_text, chosen)
            plan["request_text"] = build_schedule_request_with_llm(
                user_text=user_text,
                chosen=chosen,
                call_llm_fn=None,
            )
            start_rss_schedule(
                plan=plan,
                channel_id=event.get("channel"),
                client=client,
                say=say,
            )
        except Exception as e:
            say(text=f"⚠️ 주기 수집 예약 실패: {e}")
        return

    if tool == "get_recent_news_for_quiz":
        category = str(chosen.get("category") or "전체")
        limit = _normalize_positive_int(chosen.get("limit"), default=3) or 3
        say(text=f"📰 `{category}` 뉴스를 불러와 퀴즈를 만들고 있어요...")
        raw_res = mcp_tools.get_recent_news_for_quiz(
            category=category,
            limit=limit,
            sqlite_path=DEFAULT_SQLITE_PATH,
        )
        bundle = json.loads(raw_res)
        articles = bundle.get("articles") or []
        news_context = (
            "\n".join(
                f"- [{a.get('source') or '언론'}] {a.get('title')} | 사실: {a.get('facts')} | 관점({a.get('stance_type')}): {a.get('opinion')}"
                for a in articles
            )
            or "(조회된 최근 뉴스가 없습니다.)"
        )
        prompt = (
            f"사용자 요청: {user_text}\n\n"
            f"참고 뉴스:\n{news_context}\n\n"
            "위 뉴스를 바탕으로 4지선다형 시사 퀴즈 1문항, 정답, 해설을 만들어주세요."
        )
        say(text=ask_llm_for_quiz(prompt))
        return

    if tool == "export_sqlite_file":
        raw_res = mcp_tools.export_sqlite_file(sqlite_path=DEFAULT_SQLITE_PATH)
        info = json.loads(raw_res)
        if info.get("status") != "ok":
            say(text=f"❌ {info.get('message') or 'SQLite 파일을 찾지 못했어요.'}")
            return

        file_path = info["file_path"]
        filename = info.get("filename") or "news_store.sqlite"
        size_kb = round((info.get("size_bytes") or 0) / 1024, 1)
        channel_id = event.get("channel")

        try:
            client.files_upload_v2(
                channel=channel_id,
                file=file_path,
                filename=filename,
                title=filename,
                initial_comment=f"📦 `{filename}` ({size_kb} KB) 파일을 추출했어요. 클릭해서 다운로드하세요.",
            )
        except Exception as e:
            say(text=f"❌ Slack 파일 업로드 실패: {e}")
        return

    if tool == "get_storage_status":
        raw_res = mcp_tools.get_storage_status(sqlite_path=DEFAULT_SQLITE_PATH)
        data = json.loads(raw_res)

        active_mode = (data.get("active_backend") or "미확인").strip().lower()
        if active_mode == "sqlite":
            mode_label = f"`{active_mode}` (파일 저장중)"
        elif active_mode == "postgres":
            mode_label = f"`{active_mode}` (neon db로 저장중)"
        else:
            mode_label = f"`{active_mode}`"

        sq = data.get("sqlite") or {}
        pg = data.get("postgres") or {}

        sq_status = "정상 준비됨" if sq.get("exists") else "미준비"
        if pg.get("connected"):
            pg_status = "정상 연결됨"
        elif pg.get("configured"):
            pg_status = "연결 실패"
        else:
            pg_status = "미설정"

        say(
            text=(
                f"📌 *저장소 상태*\n"
                f"• 현재 모드: {mode_label}\n"
                f"• SQLite: {sq_status}\n"
                f"• PostgreSQL: {pg_status}"
            )
        )
        return

    if tool == "get_storage_stats":
        target_backend = _normalize_backend_value(chosen.get("target_backend"))
        raw_res = mcp_tools.get_storage_stats(
            target_backend=target_backend,
            sqlite_path=DEFAULT_SQLITE_PATH,
        )
        data = json.loads(raw_res)

        backend = (data.get("backend") or "미확인").strip().lower()
        backend_label = (
            f"`{backend}` (파일 저장중)"
            if backend == "sqlite"
            else f"`{backend}` (neon db로 저장중)"
            if backend == "postgres"
            else f"`{backend}`"
        )

        if data.get("status") != "ok":
            say(
                text=(
                    f"📈 *저장소 통계*\n"
                    f"• 대상 저장소: {backend_label}\n"
                    f"• 상태: {data.get('message') or '통계를 불러올 수 없어요.'}"
                )
            )
            return

        by_source = data.get("by_source") or []
        by_category = data.get("by_category") or []

        src_text = (
            ", ".join(f"{x['name']} {x['count']}건" for x in by_source)
            if by_source
            else "데이터 없음"
        )
        cat_text = (
            ", ".join(f"{x['name']} {x['count']}건" for x in by_category)
            if by_category
            else "데이터 없음"
        )
        oldest = data.get("oldest_published_at") or "없음"
        latest = data.get("latest_published_at") or "없음"
        period_text = (
            f"`{oldest}` ~ `{latest}`"
            if oldest and latest
            else f"`{latest}`"
            if latest
            else f"`{oldest}`"
            if oldest
            else "없음"
        )

        say(
            text=(
                f"📈 *저장소 통계*\n"
                f"• 대상 저장소: {backend_label}\n"
                f"• 총 기사 수: *{data.get('total_count', 0)}건*\n"
                f"• 언론사별(상위): {src_text}\n"
                f"• 분야별(상위): {cat_text}\n"
                f"• 기사 기간: {period_text}"
            )
        )
        return

    if tool == "sync_sqlite_to_postgres":
        sync_limit = _normalize_positive_int(chosen.get("limit"), default=None)
        raw_res = mcp_tools.sync_sqlite_to_postgres(
            limit=sync_limit,
            sqlite_path=DEFAULT_SQLITE_PATH,
        )
        data = json.loads(raw_res)
        say(text=_format_sync_result(data, "SQLite → Neon DB"))
        return

    if tool == "sync_postgres_to_sqlite":
        sync_limit = _normalize_positive_int(chosen.get("limit"), default=None)
        raw_res = mcp_tools.sync_postgres_to_sqlite(
            limit=sync_limit,
            sqlite_path=DEFAULT_SQLITE_PATH,
        )
        data = json.loads(raw_res)
        say(text=_format_sync_result(data, "Neon DB → SQLite"))
        return

    if tool == "delete_news_records":
        target_backend = _normalize_backend_value(chosen.get("target_backend"))
        source_val = (chosen.get("source") or "").strip() or None
        category_val = (chosen.get("category") or "").strip() or None
        keyword_val = (chosen.get("keyword") or "").strip() or None
        days_val = _normalize_positive_int(chosen.get("days"), default=None)
        limit_val = _normalize_positive_int(chosen.get("limit"), default=None)

        raw_order = str(chosen.get("order") or "").strip().lower()
        if raw_order in ("oldest", "asc", "오래된", "과거") or any(
            k in user_text for k in ("오래된", "옛날", "가장 먼저", "먼저 들어온")
        ):
            order_val = "oldest"
        else:
            order_val = "newest"

        delete_all_val = bool(chosen.get("delete_all") is True)

        raw_res = mcp_tools.delete_news_records(
            source=source_val,
            category=category_val,
            keyword=keyword_val,
            days=days_val,
            limit=limit_val,
            order=order_val,
            delete_all=delete_all_val,
            target_backend=target_backend,
            sqlite_path=DEFAULT_SQLITE_PATH,
        )
        data = json.loads(raw_res)
        if data.get("status") != "ok":
            say(text=f"⚠️ 삭제 처리 안내: {data.get('message') or '삭제할 수 없었어요.'}")
            return

        applied = data.get("filters") or {}
        cond_parts = []
        if applied.get("delete_all"):
            cond_parts.append("전체 데이터")
        if source_val:
            cond_parts.append(f"언론사={source_val}")
        if category_val:
            cond_parts.append(f"분야={category_val}")
        if keyword_val:
            cond_parts.append(f"키워드={keyword_val}")
        if days_val:
            cond_parts.append(f"최근 {days_val}일")
        if limit_val:
            order_label = "오래된" if order_val == "oldest" else "최근"
            cond_parts.append(f"{order_label} {limit_val}건")

        cond_summary = ", ".join(cond_parts) if cond_parts else "지정 조건"
        backend_name = data.get("backend") or "알 수 없음"
        deleted_cnt = int(data.get("deleted_count") or 0)
        say(
            text=(
                f"🗑️ *뉴스 삭제 완료* (`{backend_name}`)\n"
                f"• 적용 조건: {cond_summary}\n"
                f"• 삭제된 기사 수: *{deleted_cnt}건*"
            )
        )
        return


@app.event("message")
def handle_message_events(event, say):
    if event.get("subtype") == "bot_message" or event.get("bot_id"):
        return

    raw_text = (event.get("text") or "").strip()
    if re.search(r"<@[A-Z0-9_]+>", raw_text):
        return

    if raw_text.lower() == "!stop":
        _handle_stop_command(say)
        return

    attached_files = event.get("files") or []
    if attached_files:
        _process_slack_files(attached_files, say)


if __name__ == "__main__":
    current_backend = get_active_storage_backend()
    if current_backend == "postgres" and os.getenv("DATABASE_URL"):
        from database import init_db

        init_db()
    elif current_backend == "sqlite":
        init_sqlite_db(DEFAULT_SQLITE_PATH)

    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()