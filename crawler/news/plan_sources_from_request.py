import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

try:
    from crawler.news.rss_sources import build_sources_config, load_press_rss_map
    from crawler.news.storage_mode_config import (
        is_mode_only_request,
        resolve_storage_mode_for_request,
    )
except ImportError:
    from rss_sources import build_sources_config, load_press_rss_map
    from storage_mode_config import (
        is_mode_only_request,
        resolve_storage_mode_for_request,
    )


PRESS_SYNONYMS: Dict[str, List[str]] = {
    "연합": ["연합", "연합뉴스", "연합뉴스tv", "yonhap"],
    "경향": ["경향", "경향신문", "khan"],
    "한경": ["한경", "한국경제", "hankyung"],
    "매경": ["매경", "매일경제", "mk"],
    "조선": ["조선", "조선일보", "chosun"],
    "조선일보": ["조선일보", "조선", "chosun"],
    "조선비즈": ["조선비즈", "조선비즈/전체", "비즈", "biz"],
    "조세일보": ["조세일보", "조세", "세무"],
    "동아": ["동아", "동아일보", "donga"],
    "동아일보": ["동아일보", "동아", "donga"],
    "한겨레": ["한겨레", "한겨레신문", "hani"],
    "sbs": ["sbs", "에스비에스", "sbs뉴스"],
    "뉴시스": ["뉴시스", "newsis"],
    "aitimes": ["aitimes", "ai타임스", "에이아이타임스"],
    "전자신문": ["전자신문", "etnews", "전자"],
    "테크m": ["테크m", "테크엠", "techm"],
}

CATEGORY_SYNONYMS: Dict[str, List[str]] = {
    "종합": ["종합", "전체", "헤드라인", "메인"],
    "경제": ["경제", "금융", "증시", "거시", "경기"],
    "정치": ["정치", "국회", "대통령", "외교", "여야", "선거"],
    "사회": ["사회", "사건", "사고", "교육", "노동", "복지", "지역"],
    "세계": ["세계", "국제", "해외", "글로벌", "외신"],
    "산업_it": ["산업_it", "it/산업", "산업", "it", "테크", "기술"],
    "it_과학": ["it_과학", "it/과학", "과학", "it", "테크", "연구"],
    "it": ["it", "it/테크", "테크", "기술", "모바일"],
    "기업_경영": ["기업_경영", "기업/산업", "기업", "경영", "재계", "인사"],
    "ai/테크": ["ai/테크", "ai", "인공지능", "테크", "llm", "모델"],
    "경제/산업": ["경제/산업", "경제", "산업", "비즈"],
    "경제/세무": ["경제/세무", "세무", "조세", "세금", "경제"],
    "속보": ["속보", "긴급", "실시간"],
}


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


def _infer_default_limit(text: str) -> int:
    if any(k in text for k in ("가볍게", "몇 개", "살짝", "테스트", "빠르게", "요약만")):
        return 3
    if any(k in text for k in ("넉넉히", "많이", "듬뿍")):
        return 15
    return 5


def _build_dynamic_index(
    press_map: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, List[str], List[str]]]:
    """
    press_map의 각 항목을 (source_name, category_label, press_tokens, cat_tokens) 형태로 인덱싱합니다.
    """
    indexed: List[Tuple[str, str, List[str], List[str]]] = []
    for src_name, info in press_map.items():
        cat_label = str(info.get("category") or "종합").strip()
        press_part = src_name.split("/", 1)[0].strip() if "/" in src_name else src_name.strip()
        sec_part = src_name.split("/", 1)[1].strip() if "/" in src_name else ""

        p_tokens = {src_name.lower(), press_part.lower()}
        for syn_key, syn_list in PRESS_SYNONYMS.items():
            lower_syns = [s.lower() for s in syn_list]
            if press_part.lower() in lower_syns or src_name.lower() in lower_syns or syn_key.lower() == press_part.lower():
                p_tokens.update(lower_syns)

        c_tokens = {cat_label.lower()}
        if sec_part:
            c_tokens.add(sec_part.lower())
            c_tokens.add(sec_part.replace("_", "/").lower())
            c_tokens.add(sec_part.replace("_", " ").lower())
            c_tokens.add(sec_part.split("_", 1)[0].lower())

        for raw_c in (cat_label.lower(), sec_part.lower()):
            if not raw_c:
                continue
            if raw_c in CATEGORY_SYNONYMS:
                c_tokens.update(s.lower() for s in CATEGORY_SYNONYMS[raw_c])
            for syn_key, syn_list in CATEGORY_SYNONYMS.items():
                lower_syns = [s.lower() for s in syn_list]
                if raw_c in lower_syns or syn_key.lower() == raw_c:
                    c_tokens.update(lower_syns)

        indexed.append((
            src_name,
            cat_label,
            [t for t in p_tokens if t],
            [t for t in c_tokens if t],
        ))
    return indexed


def _fallback_parse_request(
    request_text: str,
    press_map: Dict[str, Dict[str, Any]],
    effective_mode: str,
    sqlite_path: Optional[str] = None,
) -> Dict[str, Any]:
    text = (request_text or "").strip()
    base_default_limit = _infer_default_limit(text)

    if any(kw in text for kw in ("전체 언론사", "모든 언론사", "전부 수집", "전체 수집")):
        num_match = re.search(r"(\d+)\s*개", text)
        limit = int(num_match.group(1)) if num_match else base_default_limit
        configs = build_sources_config(
            press_names=None,
            limit=limit,
            storage_mode=effective_mode,
            sqlite_path=sqlite_path,
        )
        return {"storage_mode": effective_mode, "default_limit": limit, "configs": configs}

    indexed = _build_dynamic_index(press_map)
    segments = [seg.strip() for seg in re.split(r"[,，\n]+", text) if seg.strip()]
    configs: List[Dict[str, Any]] = []
    seen_sources = set()

    for seg in segments:
        seg_lower = seg.lower()
        num_match = re.search(r"(\d+)\s*개", seg)
        seg_limit = int(num_match.group(1)) if num_match else base_default_limit

        matched_sources: List[str] = []

        # 1) 세그먼트 내 언론사 토큰과 분야/카테고리 토큰이 함께 일치하는 항목 우선 매칭
        for src_name, _, p_tokens, c_tokens in indexed:
            has_press = any(pt in seg_lower for pt in p_tokens)
            has_cat = any(ct in seg_lower for ct in c_tokens)
            if has_press and has_cat:
                matched_sources.append(src_name)

        # 2) 언론사만 언급된 경우: 해당 언론사의 대표(종합/전체/첫 번째) 항목 선택
        if not matched_sources:
            for src_name, _, p_tokens, _ in indexed:
                if any(pt in seg_lower for pt in p_tokens):
                    matched_sources.append(src_name)
                    break

        # 3) 언론사명 없이 분야/카테고리 키워드만 언급된 경우
        if not matched_sources:
            for src_name, _, _, c_tokens in indexed:
                if any(ct in seg_lower for ct in c_tokens):
                    matched_sources.append(src_name)

        for mk in matched_sources:
            if mk not in seen_sources:
                seen_sources.add(mk)
                configs.extend(
                    build_sources_config(
                        press_names=[mk],
                        limit=seg_limit,
                        storage_mode=effective_mode,
                        sqlite_path=sqlite_path,
                    )
                )

    default_limit = configs[0]["limit"] if configs else base_default_limit
    return {
        "storage_mode": effective_mode,
        "default_limit": default_limit,
        "configs": configs,
    }


def plan_sources_from_request(
    request_text: str,
    storage_mode: Optional[str] = None,
    sqlite_path: Optional[str] = None,
) -> Dict[str, Any]:
    effective_mode = resolve_storage_mode_for_request(
        request_text=request_text,
        explicit_storage_mode=storage_mode,
        persist_command=True,
    )
    if is_help_request(request_text):
        return {
            "help": True,
            "guide": get_rss_crawling_guide(),
            "configs": [],
        }

    if is_mode_only_request(request_text):
        return {
            "storage_mode": effective_mode,
            "default_limit": _infer_default_limit(request_text or ""),
            "mode_updated": True,
            "configs": [],
        }

    press_map = load_press_rss_map()
    available_summary = [
        {
            "source_name": name,
            "category": info.get("category", "종합"),
            "urls": info.get("urls", []),
        }
        for name, info in press_map.items()
    ]

    system_prompt = (
        "당신은 RSS 뉴스 수집 계획 수립기입니다. 제공된 언론사 목록의 'source_name'과 'category'를 참고해 "
        "사용자의 자연어 요청을 해석하고 유효한 JSON만 출력하세요.\n"
        "출력 형식:\n"
        "{\n"
        '  "default_limit": 5,\n'
        '  "configs": [\n'
        '    {"source_name": "목록의 정확한 source_name", "category": "카테고리", "rss_urls": ["https://..."], "limit": 4}\n'
        "  ]\n"
        "}\n"
        "규칙:\n"
        "- 반드시 제공된 목록의 정확한 source_name만 사용하세요 (예: '연합/국제' -> '연합/세계', '조선일보 정치' -> '조선/정치', '조선일보 국제' -> '조선/국제', '동아일보 경제' -> '동아일보/경제', '동아 전체' -> '동아/전체').\n"
        "- 특정 언론사나 분야가 지정되면 관련된 항목만 선택하세요. '모든 언론사/전체 언론사'처럼 명시할 때만 전체를 포함하세요.\n"
        "- 요청 문맥상 개수가 없으면 가벼운 확인/요약은 3~5, 일반 수집은 5~10 범위로 합리적으로 정하세요.\n"
        "- 언론사·분야별 개수가 다르면 각 config의 limit을 다르게 지정하세요."
    )

    user_prompt = (
        f"등록된 언론사-RSS 목록:\n{json.dumps(available_summary, ensure_ascii=False, indent=2)}\n\n"
        f"사용자 요청: {request_text}"
    )

    parsed: Optional[Dict[str, Any]] = None
    plan_max_tokens = int(os.getenv("PLAN_MAX_TOKENS", "600"))

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_groq import ChatGroq

        for model_name in _get_groq_model_queue():
            try:
                llm = ChatGroq(
                    model=model_name,
                    api_key=groq_api_key,
                    temperature=0.1,
                    max_tokens=plan_max_tokens,
                )
                resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
                raw = resp.content if isinstance(resp.content, str) else str(resp.content)
                parsed = _extract_json_block(raw)
                if parsed and isinstance(parsed.get("configs"), list):
                    break
            except Exception as e:
                print(f"[!] plan_sources Groq({model_name}) 실패: {e}", file=sys.stderr)

    if (not parsed or not isinstance(parsed.get("configs"), list)) and os.getenv("USE_OLLAMA", "false").lower() == "true":
        try:
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
            ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
            r = requests.post(
                ollama_url,
                json={"model": ollama_model, "prompt": f"{system_prompt}\n\n{user_prompt}", "stream": False},
                timeout=30,
            )
            r.raise_for_status()
            raw = r.json().get("response", "")
            parsed = _extract_json_block(raw)
        except Exception as e:
            print(f"[!] plan_sources Ollama 실패: {e}", file=sys.stderr)

    if not parsed or not isinstance(parsed.get("configs"), list):
        return _fallback_parse_request(
            request_text=request_text,
            press_map=press_map,
            effective_mode=effective_mode,
            sqlite_path=sqlite_path,
        )

    final_configs: List[Dict[str, Any]] = []
    for item in parsed.get("configs") or []:
        if not isinstance(item, dict):
            continue
        src_name = str(item.get("source_name") or "").strip()
        base_entry = press_map.get(src_name) or next(
            (v for k, v in press_map.items() if k.lower() == src_name.lower()),
            {},
        )
        urls = item.get("rss_urls") or base_entry.get("urls") or []
        if isinstance(urls, str):
            urls = [urls]
        urls = [str(u).strip() for u in urls if str(u).strip()]
        if not src_name or not urls:
            continue

        cfg: Dict[str, Any] = {
            "source_name": src_name,
            "category": str(item.get("category") or base_entry.get("category") or "종합").strip(),
            "rss_urls": urls,
            "storage_mode": effective_mode,
        }
        item_limit = item.get("limit") if item.get("limit") is not None else parsed.get("default_limit")
        if item_limit is not None:
            cfg["limit"] = int(item_limit)
        if sqlite_path:
            cfg["sqlite_path"] = sqlite_path

        final_configs.append(cfg)

    if not final_configs:
        return _fallback_parse_request(
            request_text=request_text,
            press_map=press_map,
            effective_mode=effective_mode,
            sqlite_path=sqlite_path,
        )

    return {
        "storage_mode": effective_mode,
        "default_limit": parsed.get("default_limit"),
        "configs": final_configs,
    }


def execute_rss_request(
    request_text: str,
    storage_mode: Optional[str] = None,
    sqlite_path: Optional[str] = None,
) -> Dict[str, Any]:
    """계획 수립 후 rss_base_crawler의 crawl_multiple_rss_sources로 실제 수집까지 한 번에 실행합니다."""
    try:
        from crawler.rss_base_crawler import crawl_multiple_rss_sources
    except ImportError:
        from rss_base_crawler import crawl_multiple_rss_sources

    planned = plan_sources_from_request(
        request_text=request_text,
        storage_mode=storage_mode,
        sqlite_path=sqlite_path,
    )
    configs = planned.get("configs") or []
    effective_mode = planned.get("storage_mode")

    processed = (
        crawl_multiple_rss_sources(
            sources_config=configs,
            limit=planned.get("default_limit"),
            storage_mode=effective_mode,
            sqlite_path=sqlite_path,
        )
        if configs
        else 0
    )

    return {
        "planned": planned,
        "processed_total": processed,
    }

def is_help_request(text: Optional[str]) -> bool:
    if not text:
        return False
    cleaned = text.strip().lower()
    return cleaned in ("--help", "-h", "help", "도움말", "가이드") or "--help" in cleaned


def get_rss_crawling_guide() -> str:
    return """
=== RSS 뉴스 크롤링 가이드 ===

1. 자연어 수집 요청 예시
   - "연합/국제 4개, 경향에서 종합 2개, 조선일보 정치 3개 모아줘"
   - "동아일보 경제 4개, AI 뉴스 2개 가볍게 가져와줘"
   - "모든 국제 뉴스 가볍게 5개 모아줘"

2. 저장소 모드 전환 (서버 상태 파일 storage_mode_state.json에 저장)
   - "앞으로 sqlite로 저장해줘"  -> 이후 요청 기본값을 sqlite로 유지
   - "다시 db로 바꿔줘"          -> 이후 요청 기본값을 postgres(DB)로 복귀
   - 호출 시 storage_mode="sqlite" 또는 "postgres"를 직접 넘겨 덮어쓸 수도 있습니다.

3. 주요 등록 소스(source_name) 예시
   - 연합: 연합/종합, 연합/경제, 연합/산업_IT, 연합/정치, 연합/사회, 연합/세계
   - 경향: 경향/종합, 경향/경제, 경향/정치, 경향/사회, 경향/IT_과학
   - 경제/IT: 한경/전체, 한경/경제, 한경/IT, 매경/경제, 매경/기업_경영, AITIMES, 전자신문/전체, 테크M
   - 일간지/기타: 한겨레/전체, 조선일보/전체, 조선/정치, 조선/국제, 조선비즈/전체, 동아/전체, 동아일보/경제, SBS/뉴스, 뉴시스/속보, 조세일보/전체

4. 팁
   - 개수를 따로 적지 않으면 문맥("가볍게/몇 개/테스트" -> 3개, 일반 -> 5개, "넉넉히/많이" -> 15개)에 맞춰 자동 설정됩니다.
   - "연합/국제", "조선일보 정치"처럼 자연어 별칭을 써도 등록된 source_name으로 자동 매칭됩니다.
""".strip()


