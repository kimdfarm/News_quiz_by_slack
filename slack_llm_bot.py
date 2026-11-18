import importlib.util
import json
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

_CRAWL_LOCK = threading.Lock()
_IS_CRAWLING = False


def is_crawling() -> bool:
    with _CRAWL_LOCK:
        return _IS_CRAWLING


def set_crawling(active: bool) -> None:
    global _IS_CRAWLING
    with _CRAWL_LOCK:
        _IS_CRAWLING = active


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MCP_DIR = os.path.join(_BASE_DIR, "mcp")

for _p in (_BASE_DIR, _MCP_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

_SERVER_PATH = os.path.join(_MCP_DIR, "news_mcp_server.py")
_spec = importlib.util.spec_from_file_location("news_mcp_server", _SERVER_PATH)
mcp_tools = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(mcp_tools)

from crawler.news.plan_sources_from_request import (
    execute_rss_request,
    get_rss_crawling_guide,
    parse_help_topic,
)
from storage_backend import (
    DEFAULT_SQLITE_PATH,
    get_active_storage_backend,
)

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# 추후 Tool RAG / 검색 인덱스로 확장할 때 그대로 재사용할 카탈로그
TOOL_GROUPS: Dict[str, List[Dict[str, str]]] = {
    "storage": [
        {
            "tool": "set_storage_backend",
            "description": (
                "저장소 백엔드 모드(sqlite 파일 vs postgres/neon db)를 변경한다. "
                "예: '네온db로 변경해줘', 'sqlite로 바꿔줘', '저장 상태를 파일 형태로 바꿔줘'"
            ),
        },
        {
            "tool": "export_sqlite_file",
            "description": (
                "로컬 SQLite DB 파일(news_store.sqlite) 자체를 Slack으로 추출/다운로드한다. "
                "예: 'sqlite 파일 추출해줘', 'news_store.sqlite 다운로드', 'sqlite 파일 보내줘'"
            ),
        },
        {
            "tool": "get_storage_status",
            "description": (
                "현재 활성 저장소 모드와 SQLite / Postgres 연결·파일 상태를 조회한다. "
                "예: '저장소 상태 확인해줘', 'DB 연결 상태 보고해줘', '현재 어디에 저장 중이지?'"
            ),
        },
        {
            "tool": "get_storage_stats",
            "description": (
                "저장소(sqlite 또는 postgres)의 기사 수, 언론사별·분야별·기간 통계를 조회한다. "
                "예: '저장된 정보 통계 보여줘', '몇 건 쌓였어?', '통계 알려줘'"
            ),
        },
        {
            "tool": "sync_sqlite_to_postgres",
            "description": (
                "[단방향: SQLite 파일 -> Neon DB(PostgreSQL)] 로컬 파일 데이터를 디지털 DB(Neon/Postgres)로 올린다. "
                "예: 'sqlite를 postgres로 동기화해줘', '로컬 데이터 네온으로 올려줘', '파일 데이터 디지털 DB로 동기화'"
            ),
        },
        {
            "tool": "sync_postgres_to_sqlite",
            "description": (
                "[단방향: Neon DB(PostgreSQL) -> SQLite 파일] 디지털 DB(Neon/Postgres) 데이터를 로컬 파일로 내려받는다. "
                "예: 'postgres를 sqlite로 동기화해줘', '네온 db 데이터를 로컬 파일로 가져와줘', '디지털 DB 데이터를 파일로 동기화'"
            ),
        },
    ],
    "content": [
        {
            "tool": "run_rss_by_request",
            "description": (
                "자연어 요청을 해석해 RSS 뉴스를 크롤링/수집한다. "
                "예: '국제 뉴스 3개 모아줘', '연합/경제 수집'"
            ),
        },
        {
            "tool": "get_recent_news_for_quiz",
            "description": (
                "저장된 최근 뉴스를 바탕으로 시사 퀴즈를 생성한다. "
                "예: '경제 뉴스로 퀴즈 내줘', '최근 뉴스 문제 만들어줘'"
            ),
        },
    ],
}

VALID_TOOLS = {
    item["tool"]
    for tools in TOOL_GROUPS.values()
    for item in tools
} | {"none"}


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
    """LLM이 반환한 저장소 명칭을 'sqlite' 또는 'postgres'로 정규화(불명확 시 None)."""
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
    """LLM이 추출한 limit/정수 값을 안전하게 양의 정수로 변환."""
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


def _is_ollama_model_available(model_name: str, base_url: str, timeout: float = 1.5) -> bool:
    """로컬 Ollama에 해당 모델이 있는지 확인하고, 없으면 가동하지 않습니다."""
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
    """
    LangChain 기반 상용(Groq) / 로컬(Ollama) 호출기.
    - prefer_local=False: Groq 우선 시도 -> 실패 시 Ollama 모델 존재 여부 확인 후 폴백(없으면 미가동)
    - prefer_local=True: Ollama 모델 존재 여부 확인 후 호출 -> 없으면 Groq 시도
    """
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


def _route_tool_group(user_text: str) -> str:
    """1단계 라우터: 요청이 속하는 기능 그룹만 LLM으로 분류."""
    router_prompt = (
        "사용자 요청이 속하는 기능 그룹 하나만 문맥으로 판별하세요.\n"
        "- storage: 저장소 모드 변경, 상태/통계 조회, sqlite 파일 추출, 저장소 간 데이터 동기화\n"
        "- content: RSS 뉴스 수집/크롤링, 저장된 뉴스 기반 시사 퀴즈 생성\n"
        "- none: 위 그룹에 해당하지 않는 요청\n\n"
        '설명 없이 순수 JSON 1개만 출력하세요: {"group": "storage | content | none"}'
    )
    raw = _invoke_llm_with_fallback(
        messages=[SystemMessage(content=router_prompt), HumanMessage(content=user_text)],
        temperature=0.0,
        max_tokens=120,
        prefer_local=False,
    )
    parsed = _extract_json(raw or "") or {}
    group = str(parsed.get("group") or "none").strip().lower()
    print(f"[DEBUG] text={user_text!r} -> parsed={parsed}", file=sys.stderr)
    return group if group in ("storage", "content", "none") else "none"


def _choose_mcp_tool(user_text: str) -> Dict[str, Any]:
    """2단계 선택기: 라우터가 고른 그룹의 도구 목록을 바탕으로 도구와 파라미터를 LLM이 직접 해석."""
    group = _route_tool_group(user_text)
    if group == "none":
        return {"tool": "none"}

    candidates = TOOL_GROUPS[group]
    if group == "storage":
        system_prompt = (
            "당신은 저장소 및 동기화 MCP 도구 선택기입니다. 사용자 발화의 의도, 출발지/목적지 방향, "
            "대상 저장소, 건수 제한(limit)을 문맥으로 파악해 순수 JSON 1개만 출력하세요.\n"
            f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
            "판별 기준:\n"
            "- 출발지가 로컬/파일/SQLite이고 목적지가 Neon/Postgres/디지털 DB/DB인 동기화 -> sync_sqlite_to_postgres\n"
            "- 출발지가 Neon/Postgres/디지털 DB/DB이고 목적지가 로컬/파일/SQLite인 동기화 -> sync_postgres_to_sqlite\n"
            "- 저장소 모드/백엔드 자체를 변경 -> set_storage_backend (storage_backend에 'sqlite' 또는 'postgres')\n"
            "- SQLite 파일 자체를 내보내기/다운로드 -> export_sqlite_file\n"
            "- 현재 저장소 설정/연결/파일 상태 확인 -> get_storage_status\n"
            "- 기사 건수/언론사/분야/기간 통계 조회 -> get_storage_stats (특정 저장소가 언급되면 target_backend에 'sqlite' 또는 'postgres', 없으면 null)\n"
            "- 해당 없음 -> none\n\n"
            "limit 규칙:\n"
            "- 특정 건수/개수 제한이 언급되면 양의 정수로 넣고, 전체/모두/다 또는 건수 제한 언급이 없으면 null을 넣으세요.\n\n"
            '출력 형식: {"tool": "...", "storage_backend": null, "target_backend": null, "limit": null}'
        )
    else:
        system_prompt = (
            "당신은 뉴스 수집 및 퀴즈 MCP 도구 선택기입니다. 사용자 발화를 문맥으로 파악해 "
            "순수 JSON 1개만 출력하세요.\n"
            f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
            '출력 형식: {"tool": "run_rss_by_request | get_recent_news_for_quiz | none", "category": "전체", "limit": 3}'
        )

    raw = _invoke_llm_with_fallback(
        messages=[SystemMessage(content=system_prompt), HumanMessage(content=user_text)],
        temperature=0.0,
        max_tokens=300,
        prefer_local=False,
    )
    parsed = _extract_json(raw or "")
    if parsed and parsed.get("tool") in VALID_TOOLS:
        return parsed
    print(f"[DEBUG] text={user_text!r} -> parsed={parsed}", file=sys.stderr)
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


@app.event("app_mention")
def handle_mention(event, say, client):
    if is_crawling():
        say(text="현재 다른 사용자가 뉴스를 수집하고 있어요. 잠시 후 다시 시도해 주세요.")
        return
    user_text = _clean_mention_text(event.get("text", ""))

    help_topic = parse_help_topic(user_text)
    if help_topic:
        say(text=get_rss_crawling_guide(help_topic))
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
            target_mode = (
                "postgres" if get_active_storage_backend() == "sqlite" else "sqlite"
            )
        raw_res = mcp_tools.set_storage_backend(target_mode)
        data = json.loads(raw_res)
        say(text=f"✅ 저장소 백엔드를 `{data.get('STORAGE_BACKEND')}`로 변경했어요.")
        return

    if tool == "run_rss_by_request":
        if is_crawling():
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

    if tool == "get_recent_news_for_quiz":
        category = str(chosen.get("category") or "전체")
        limit = _normalize_positive_int(chosen.get("limit"), default=3) or 3
        say(text=f"📰 `{category}` 뉴스를 불러와 퀴즈를 만들고 있어요...")
        raw_res = mcp_tools.get_recent_news_for_quiz(category=category, limit=limit)
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
        if oldest and latest:
            period_text = f"`{oldest}` ~ `{latest}`"
        elif latest:
            period_text = f"`{latest}`"
        elif oldest:
            period_text = f"`{oldest}`"
        else:
            period_text = "없음"

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

    say(text="그런 기능은 없어요.")


if __name__ == "__main__":
    current_backend = get_active_storage_backend()
    if current_backend == "postgres" and os.getenv("DATABASE_URL"):
        from database import init_db
        init_db()
    elif current_backend == "sqlite":
        from crawler.sqlite_store import init_sqlite_db
        init_sqlite_db()

    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()