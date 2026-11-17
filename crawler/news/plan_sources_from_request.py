import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import requests

load_dotenv()

try:
    from crawler.news.rss_sources import load_press_rss_map
    from crawler.news.storage_mode_config import (
        is_mode_only_request,
        resolve_storage_mode_for_request,
    )
except ImportError:
    from rss_sources import load_press_rss_map
    from storage_mode_config import (
        is_mode_only_request,
        resolve_storage_mode_for_request,
    )


def parse_help_topic(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    cleaned = text.strip().lower()
    if "--help" not in cleaned and cleaned not in ("-h", "help", "도움말", "가이드"):
        return None
    if re.search(r"\b--?incoder\b", cleaned) or "incoder" in cleaned:
        return "incoder"
    if re.search(r"\b--?decoder\b", cleaned) or "decoder" in cleaned:
        return "decoder"
    return "index"


def get_rss_crawling_guide(topic: Optional[str] = "index") -> str:
    target = (topic or "index").strip().lower()
    if target == "incoder":
        return (
            "*[--incoder --help] RSS 데이터 추출 및 저장 모드 도움말*\n"
            "• 자연어로 언론사/분야/개수를 적으면 등록된 RSS 목록 기준으로 추론 수집합니다.\n"
            "• 저장소 전환: `앞으로 sqlite로 저장해줘` / `다시 db로 바꿔줘` (`storage_mode_state.json`에 저장)\n"
            "• 등록된 소스 중 일부가 없어도 가능한 항목만 수집하고 미지원 항목을 함께 안내합니다."
        )
    if target == "decoder":
        return (
            "*[--decoder --help] 문제 템플릿 출력 방식 가이드*\n"
            "• 저장된 뉴스의 사실·관점·키워드 데이터를 바탕으로 퀴즈 템플릿을 생성·출력합니다."
        )
    return (
        "*Slack 봇 도움말*\n"
        "• `--incoder --help`  # RSS 데이터 추출 방법 및 저장 모드 도움말\n"
        "• `--decoder --help`  # 문제 템플릿 출력 방식 도움말"
    )


def _get_groq_model_queue() -> List[str]:
    raw_list = os.getenv("GROQ_MODELS", "").strip()
    if raw_list:
        models = [m.strip() for m in raw_list.split(",") if m.strip()]
        if models:
            return list(dict.fromkeys(models))

    primary = (os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b").strip()
    fallbacks_raw = os.getenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b").strip()
    fallbacks = [m.strip() for m in fallbacks_raw.split(",") if m.strip()]
    return list(dict.fromkeys([primary] + fallbacks))


def _extract_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


def _extract_explicit_limit(request_text: str) -> Optional[int]:
    if not request_text:
        return None
    m = re.search(r"(?:각\s*)?(\d+)\s*(?:개|건|기사|뉴스)", request_text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 50:
            return val
    return None


from langchain_ollama import ChatOllama


def _is_ollama_model_available(model_name: str, base_url: str, timeout: float = 1.5) -> bool:
    """로컬 Ollama 서버에 해당 모델이 다운로드되어 있는지 확인합니다."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return False
        payload = resp.json() if resp.content else {}
        models = payload.get("models") or []
        target = model_name.strip().lower()
        for m in models:
            name = str((m or {}).get("name") or (m or {}).get("model") or "").strip().lower()
            if name == target or name.startswith(f"{target}:"):
                return True
        return False
    except Exception:
        return False


def _infer_llm_plan(
    request_text: str,
    available_summary: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    catalog_lines = [
        f'- 키(source_name): "{item["source_name"]}", 분야(category): "{item["category"]}"'
        for item in available_summary
    ]
    catalog_text = "\n".join(catalog_lines)

    system_prompt = (
        "당신은 RSS 뉴스 수집 요청을 분석해 등록된 RSS 소스 목록과 유연하게 매칭하는 플래너입니다.\n"
        "제공된 '등록된 RSS 소스 목록'의 키(source_name) 이름과 분야(category)를 보고 "
        "사용자 요청 의도에 가장 알맞은 소스들을 선택해 유효한 JSON만 출력하세요.\n\n"
        "출력 스키마:\n"
        "{\n"
        '  "default_limit": 5,\n'
        '  "configs": [\n'
        '    {"source_name": "목록의 정확한 source_name", "category": "카테고리", "limit": 3}\n'
        "  ],\n"
        '  "unsupported_items": ["목록에 없어 RSS로 제공할 수 없는 요청 항목"]\n'
        "}\n\n"
        "핵심 추론 원칙:\n"
        "1. 등록된 소스 키(source_name)는 주로 '언론사/종합', '언론사/전체', '언론사/분야' 형태입니다.\n"
        "2. '전체 언론사', '전 언론사', '언론사별 전반적 뉴스'처럼 세부 섹션을 지정하지 않은 전체 요청에서는 "
        "세부 분야 피드(스포츠, 야구, 골프, 사설칼럼, 여행 등)를 무작정 다 넣기보다, "
        "키 이름에 '종합'이나 '전체'가 포함된 대표 피드(예: 연합/종합, 경향/종합, 한경/전체, 한겨레/전체, "
        "조선일보/전체, 조선비즈/전체, 조세일보/전체, 전자신문/전체, 동아/전체 등)를 중심으로 유연하게 선택하세요.\n"
        "   단, '모든 RSS 피드/세부 섹션까지 전부'처럼 명시하면 목록 전체를 선택할 수 있습니다.\n"
        "3. 특정 언론사명만 말하면(예: '조선일보', '동아') 해당 언론사의 대표/관련 키를 문맥에 맞게 고르고, "
        "특정 분야나 키워드(예: '경제', 'AI/테크', '정치', '국제', '스포츠/야구')를 말하면 "
        "키 이름과 category가 부합하는 소스들을 선택하세요.\n"
        "4. RSS 소스 목록에 없는 주제나 요청(예: 날씨, 실시간 시세, 개인 일정 등)은 억지로 매칭하지 말고 "
        "'unsupported_items'에 넣으세요. 일부만 가능하면 가능한 것은 configs에, 없는 것만 unsupported_items에 넣으세요.\n"
        "5. 요청에 개수('각 2개', '3건' 등)가 있으면 limit와 default_limit에 반영하고, 없으면 문맥에 맞게 3~10 사이로 정하세요."
    )

    user_prompt = (
        f"등록된 RSS 소스 목록:\n{catalog_text}\n\n"
        f"사용자 요청: {request_text}"
    )

    plan_max_tokens = int(os.getenv("PLAN_MAX_TOKENS", "1400"))
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()

    # 1) 상용 Groq 모델 큐 우선 실행
    if groq_api_key:
        for model_name in _get_groq_model_queue():
            try:
                llm = ChatGroq(
                    model=model_name,
                    api_key=groq_api_key,
                    temperature=0.1,
                    max_tokens=plan_max_tokens,
                )
                resp = llm.invoke(messages)
                raw = resp.content if isinstance(resp.content, str) else str(resp.content)
                parsed = _extract_json_block(raw)
                if parsed and isinstance(parsed.get("configs"), list):
                    return parsed
            except Exception as e:
                print(f"[!] plan_sources Groq({model_name}) 실패: {e}", file=sys.stderr)

    # 2) Groq가 안 될 때만 로컬 Ollama에 gemma3:4b 존재 여부 확인 후 호출
    ollama_model = (os.getenv("OLLAMA_MODEL") or "gemma3:4b").strip()
    ollama_base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()

    if not _is_ollama_model_available(ollama_model, ollama_base_url):
        print(
            f"[i] 로컬 Ollama에 '{ollama_model}' 모델이 없어 로컬 실행을 건너뜁니다.",
            file=sys.stderr,
        )
        return None

    try:
        ollama_llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.1,
        )
        resp = ollama_llm.invoke(messages)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        parsed = _extract_json_block(raw)
        if parsed and isinstance(parsed.get("configs"), list):
            return parsed
    except Exception as e:
        print(f"[!] plan_sources Ollama({ollama_model}) 호출 실패: {e}", file=sys.stderr)

    return None


def _build_status_message(configs: List[Dict[str, Any]], unsupported_items: List[str]) -> str:
    unsupported_clean = [str(x).strip() for x in (unsupported_items or []) if str(x).strip()]
    if not configs and unsupported_clean:
        return f"요청하신 항목({', '.join(unsupported_clean)})은 RSS가 제공하지 않아 수집을 실행하지 않습니다."
    if not configs:
        return "요청하신 내용에 맞는 RSS 소스를 찾지 못해 수집을 실행하지 않습니다."
    if unsupported_clean:
        return f"일부분({', '.join(unsupported_clean)})은 RSS가 제공하지 않습니다. 가능한 항목만 수집합니다."
    return ""


def plan_sources_from_request(
    request_text: str,
    storage_mode: Optional[str] = None,
    sqlite_path: Optional[str] = None,
) -> Dict[str, Any]:
    help_topic = parse_help_topic(request_text)
    if help_topic:
        return {
            "help": True,
            "topic": help_topic,
            "guide": get_rss_crawling_guide(help_topic),
            "configs": [],
            "unsupported_items": [],
            "message": "",
        }

    effective_mode = resolve_storage_mode_for_request(
        request_text=request_text,
        explicit_storage_mode=storage_mode,
        persist_command=True,
    )

    if is_mode_only_request(request_text):
        return {
            "storage_mode": effective_mode,
            "default_limit": 5,
            "mode_updated": True,
            "configs": [],
            "unsupported_items": [],
            "message": "",
        }

    press_map = load_press_rss_map()
    available_summary = [
        {
            "source_name": name,
            "category": info.get("category", "종합"),
        }
        for name, info in press_map.items()
    ]

    parsed = _infer_llm_plan(request_text=request_text, available_summary=available_summary)
    if not parsed:
        return {
            "storage_mode": effective_mode,
            "default_limit": 5,
            "configs": [],
            "unsupported_items": [request_text.strip()] if (request_text or "").strip() else [],
            "message": "요청을 추론할 수 없어 수집을 실행하지 않았습니다.",
        }

    explicit_limit = _extract_explicit_limit(request_text)
    default_limit = explicit_limit if explicit_limit is not None else int(parsed.get("default_limit") or 5)
    raw_unsupported = parsed.get("unsupported_items")
    unsupported_items = (
        [str(x).strip() for x in raw_unsupported if str(x).strip()]
        if isinstance(raw_unsupported, list)
        else []
    )

    final_configs: List[Dict[str, Any]] = []
    seen_sources = set()

    for item in parsed.get("configs") or []:
        if not isinstance(item, dict):
            continue
        src_name = str(item.get("source_name") or "").strip()
        if not src_name or src_name not in press_map or src_name in seen_sources:
            continue

        base_entry = press_map[src_name]
        urls = list(base_entry.get("urls") or [])
        if not urls:
            continue

        seen_sources.add(src_name)
        item_limit = (
            explicit_limit
            if explicit_limit is not None
            else (item.get("limit") if item.get("limit") is not None else default_limit)
        )
        cfg: Dict[str, Any] = {
            "source_name": src_name,
            "category": str(item.get("category") or base_entry.get("category") or "종합").strip(),
            "rss_urls": urls,
            "storage_mode": effective_mode,
            "limit": int(item_limit),
        }
        if sqlite_path:
            cfg["sqlite_path"] = sqlite_path
        final_configs.append(cfg)

    status_message = _build_status_message(final_configs, unsupported_items)

    return {
        "storage_mode": effective_mode,
        "default_limit": default_limit,
        "configs": final_configs,
        "unsupported_items": unsupported_items,
        "message": status_message,
    }


def execute_rss_request(
    request_text: str,
    storage_mode: Optional[str] = None,
    sqlite_path: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """LLM 추론 계획을 바탕으로 지원되는 소스만 수집하고, 미지원 항목 안내 및 불필요한 크롤링 차단을 수행합니다."""
    try:
        from crawler.rss_base_crawler import crawl_multiple_rss_sources
    except ImportError:
        from rss_base_crawler import crawl_multiple_rss_sources

    planned = plan_sources_from_request(
        request_text=request_text,
        storage_mode=storage_mode,
        sqlite_path=sqlite_path,
    )

    if planned.get("help"):
        return {
            "help": True,
            "guide": planned.get("guide", ""),
            "planned": planned,
            "processed_total": 0,
            "skipped_crawl": True,
            "message": "",
        }

    if planned.get("mode_updated"):
        return {
            "planned": planned,
            "processed_total": 0,
            "skipped_crawl": True,
            "message": f"저장 모드를 `{planned.get('storage_mode')}`로 변경했습니다.",
        }

    configs = planned.get("configs") or []
    effective_mode = planned.get("storage_mode")
    message = planned.get("message") or ""

    if not configs:
        return {
            "planned": planned,
            "processed_total": 0,
            "skipped_crawl": True,
            "message": message or "요청하신 항목은 RSS가 제공하지 않아 수집을 실행하지 않습니다.",
        }

    processed = crawl_multiple_rss_sources(
        sources_config=configs,
        limit=planned.get("default_limit"),
        storage_mode=effective_mode,
        sqlite_path=sqlite_path,
        on_progress=on_progress,
    )

    return {
        "planned": planned,
        "processed_total": processed,
        "skipped_crawl": False,
        "message": message,
    }