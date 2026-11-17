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
    normalize_storage_backend,
)

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

MCP_TOOL_CATALOG = [
    {
        "tool": "run_rss_by_request",
        "description": "자연어 요청을 해석해 RSS 뉴스를 크롤링/수집한다. (예: 국제 뉴스 3개 모아줘, 연합/경제 수집)",
    },
    {
        "tool": "set_storage_backend",
        "description": "저장소 백엔드를 postgres(네온db/DB) 또는 sqlite(로컬/파일)로 변경한다.",
    },
    {
        "tool": "export_sqlite_file",
        "description": "SQLite DB 파일(news_store.sqlite)을 Slack으로 추출/다운로드할 수 있게 업로드한다.",
    },
    {
        "tool": "get_recent_news_for_quiz",
        "description": "저장된 최근 뉴스를 조회해 시사 퀴즈를 만든다.",
    },
    {
        "tool": "get_storage_status",
        "description": "현재 저장소 설정과 SQLite / Postgres 연결·파일 상태를 조회한다.",
    },
    {
        "tool": "get_storage_stats",
        "description": "현재 또는 지정 저장소(sqlite / postgres)의 기사 수 및 언론사/분야별 통계를 조회한다.",
    },
    {
        "tool": "none",
        "description": "위 도구 목록에 없는 기능이나 무관한 요청이면 반드시 선택한다.",
    },
]

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


def _choose_mcp_tool(user_text: str) -> Dict[str, Any]:
    system_prompt = (
        "당신은 Slack 봇의 MCP 도구 선택기입니다. 사용자 요청을 보고 아래 도구 중 하나만 고르세요.\n"
        f"{json.dumps(MCP_TOOL_CATALOG, ensure_ascii=False, indent=2)}\n\n"
        "설명/잡담/마크다운 없이 순수 JSON 객체 1개만 출력하세요:\n"
        '{"tool": "run_rss_by_request | set_storage_backend | export_sqlite_file | get_recent_news_for_quiz | get_storage_status | none", '
        '"storage_backend": "sqlite | postgres | null", "category": "전체", "limit": 5}\n'
        "규칙:\n"
        "- 저장소/DB/데이터 저장 위치·모드 변경 요청(예: '네온db로 변경해줘', 'sqlite로 바꿔줘', '저장 상태를 파일 형태로 바꿔줘', '로컬 파일로 저장해줘', '디비로 변경') -> set_storage_backend\n"
        "  * sqlite/로컬/파일 저장 모드 전환 -> storage_backend: \"sqlite\"\n"
        "  * postgres/neon/네온db/db/디비 저장 모드 전환 -> storage_backend: \"postgres\"\n"
        "- 저장된 sqlite 파일 자체를 다운로드/추출/내보내기/보내줘 요청(예: 'sqlite 파일 추출해줘', 'news_store.sqlite 다운로드', 'sqlite 파일 보내줘') -> export_sqlite_file\n"
        "- RSS 수집/크롤링/기사 모아줘/기사 수집해줘 요청 -> run_rss_by_request\n"
        "- 퀴즈/문제 출제 요청 -> get_recent_news_for_quiz\n"
        "- 저장소 상태/DB 상태/SQLite 상태/연결 상태 조회 요청(예: '저장소 상태 확인해줘', 'DB 연결 상태 보고해줘', '현재 어디에 저장 중이지?') -> get_storage_status\n"
        "- 저장소 통계/기사 건수/언론사별·분야별 현황 조회 요청(예: '저장된 정보 통계 보여줘', '몇 건 쌓였어?', '통계 알려줘', '기사 수 확인해줘') -> get_storage_stats"
        "- 그 외 모든 요청 -> none"
    )

    valid_tools = {
        "run_rss_by_request",
        "set_storage_backend",
        "export_sqlite_file",
        "get_recent_news_for_quiz",
        "get_storage_status",
        "get_storage_stats",
        "none",
    }

    raw = _invoke_llm_with_fallback(
        messages=[SystemMessage(content=system_prompt), HumanMessage(content=user_text)],
        temperature=0.0,
        max_tokens=600,
        prefer_local=False,
    )
    parsed = _extract_json(raw or "")
    if parsed and parsed.get("tool") in valid_tools:
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


@app.event("app_mention")
def handle_mention(event, say,client):
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
        raw_backend = str(chosen.get("storage_backend") or "").strip().lower()
        if raw_backend in ("sqlite", "file", "local", "로컬", "파일"):
            target_mode = "sqlite"
        elif raw_backend in ("postgres", "postgresql", "db", "디비", "neon", "네온", "네온db"):
            target_mode = "postgres"
        else:
            lowered = user_text.lower()
            target_mode = (
                "sqlite"
                if any(k in lowered for k in ("sqlite", "로컬", "파일"))
                else "postgres"
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
        limit = int(chosen.get("limit") or 3)
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
        lowered = user_text.lower()
        if any(k in lowered for k in ("sqlite", "로컬", "파일")) and not any(
            k in lowered for k in ("postgres", "neon", "네온", "db", "디비")
        ):
            target_backend = "sqlite"
        elif any(k in lowered for k in ("postgres", "neon", "네온", "db", "디비")):
            target_backend = "postgres"
        else:
            target_backend = None

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