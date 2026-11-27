import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from tool_rag_router import MCP_TOOL_REGISTRY, retrieve_top_tools

VALID_TOOLS = {item["tool"] for item in MCP_TOOL_REGISTRY} | {"none"}


def extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
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


def unwrap_tool_args(chosen: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(chosen or {})
    nested = merged.get("inputs")
    if isinstance(nested, dict):
        for k, v in nested.items():
            if k not in merged or merged[k] is None:
                merged[k] = v
    return merged


def clean_str_arg(val: Any) -> Optional[str]:
    if val is None or isinstance(val, bool):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("null", "none") or "|" in s:
        return None
    return s


def to_nonneg_float(val: Any, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        num = float(str(val).strip())
        return num if num >= 0 else default
    except Exception:
        return default


def format_seconds_hms(total_sec: float) -> str:
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


def invoke_llm_with_fallback(
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


def _build_candidate_input_specs(top_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for t in top_tools:
        specs.append({
            "tool": t.get("tool"),
            "description": t.get("description", ""),
            "examples": (t.get("examples") or [])[:2],
            "inputs": t.get("schema") or {},
        })
    return specs


def choose_mcp_tool(user_text: str) -> Dict[str, Any]:
    top_tools = retrieve_top_tools(user_text, top_k=3, min_score=1.5)
    if not top_tools:
        print(f"[DEBUG] text={user_text!r} -> no matching tools (score below threshold)", file=sys.stderr)
        return {"tool": "none"}

    candidate_inputs = _build_candidate_input_specs(top_tools)
    system_prompt = (
        "당신은 사용자 요청의 최종 목적(서술어)을 파악해 후보 도구 중 가장 알맞은 JSON 1개를 출력하는 라우터입니다.\n"
        "판단 규칙:\n"
        "1. 앞부분의 수식어보다 문장의 최종 행동(예: '~로 변경/전환' vs '~파일 추출/다운로드')을 기준으로 도구를 선택하세요.\n"
        "   - 저장 방식/모드를 바꾸는 요청 -> set_storage_backend\n"
        "   - DB 파일 자체를 다운로드/업로드해 달라는 요청 -> export_sqlite_file\n"
        "2. 마크다운, 코드블록, 부가 설명 없이 오직 순수 JSON 객체 1개만 출력하세요.\n"
        "3. 후보 중 맞는 도구가 없으면 {\"tool\": \"none\"}을 출력하세요.\n\n"
        f"후보 도구 목록:\n{json.dumps(candidate_inputs, ensure_ascii=False)}"
    )

    for attempt in range(2):
        raw = invoke_llm_with_fallback(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_text),
            ],
            temperature=0.0,
            max_tokens=400,
            prefer_local=True,
        )
        parsed = extract_json(raw or "")
        print(
            f"[DEBUG] text={user_text!r} candidates={[c['tool'] for c in candidate_inputs]} "
            f"attempt={attempt+1} -> parsed={parsed}",
            file=sys.stderr,
        )
        if parsed and parsed.get("tool") in VALID_TOOLS:
            return parsed

    return {"tool": "none"}


def resolve_backend_value_with_llm(
    raw_val: Any,
    user_text: str = "",
    default: Optional[str] = None,
) -> Optional[str]:
    if raw_val is None or isinstance(raw_val, bool):
        raw_str = ""
    else:
        raw_str = str(raw_val).strip()

    if not raw_str and not (user_text or "").strip():
        return default

    prompt = (
        "당신은 저장소 백엔드 대상을 판별하는 분류기입니다.\n"
        "후보 값과 사용자 원문을 보고 아래 세 값 중 하나만 JSON으로 출력하세요:\n"
        '- "sqlite": 로컬 파일 기반 SQLite 저장소(예: sqlite, 파일, 로컬 파일, news_store.sqlite 등)\n'
        '- "postgres": 원격 DB 저장소(예: postgres, postgresql, neon, 네온, 네온db, db 등)\n'
        "- null: 특정 저장소가 명시되지 않았거나 미지정/자동/null인 경우\n\n"
        "규칙:\n"
        "1. 마크다운, 코드블록, 설명 없이 오직 JSON 1개만 출력하세요.\n"
        '출력 형식: {"backend": "sqlite"} 또는 {"backend": "postgres"} 또는 {"backend": null}\n\n'
        f"후보 값: {raw_val!r}\n"
        f"사용자 원문: {user_text!r}"
    )

    parsed = extract_json(
        invoke_llm_with_fallback(
            messages=[HumanMessage(content=prompt)],
            temperature=0.0,
            max_tokens=100,
            prefer_local=True,
        )
        or ""
    )

    if isinstance(parsed, dict) and "backend" in parsed:
        b = parsed.get("backend")
        if b is None:
            return None
        b_clean = str(b).strip().lower()
        if b_clean in ("sqlite", "postgres"):
            return b_clean

    low = f"{raw_str} {user_text}".lower()
    if "sqlite" in low:
        return "sqlite"
    if "postgres" in low or "neon" in low or "네온" in low:
        return "postgres"
    return default


def resolve_positive_int_with_llm(
    raw_val: Any,
    user_text: str = "",
    field_name: str = "count",
    default: Optional[int] = None,
) -> Optional[int]:
    if raw_val is None or isinstance(raw_val, bool):
        return default

    raw_str = str(raw_val).strip()
    if not raw_str:
        return default

    prompt = (
        "당신은 사용자 요청과 추출된 값을 보고 양의 정수(1 이상의 정수)를 확정하는 판별기입니다.\n"
        "규칙:\n"
        "1. '전체', '모두', '다', '전부', '제한없음', '미지정', 'null', 'none', 0 이하, 음수 등 "
        "구체적인 양의 개수가 아니거나 전체 대상이면 반드시 null을 출력하세요.\n"
        "2. 명확한 양의 정수(예: 1, 3, 5, '3개', '최근 7일')라면 해당 정수만 숫자로 출력하세요.\n"
        "3. 마크다운, 코드블록, 설명 없이 오직 JSON 1개만 출력하세요.\n"
        '출력 형식: {"value": 3} 또는 {"value": null}\n\n'
        f"필드명: {field_name}\n"
        f"후보 값: {raw_val!r}\n"
        f"사용자 원문: {user_text!r}"
    )

    parsed = extract_json(
        invoke_llm_with_fallback(
            messages=[HumanMessage(content=prompt)],
            temperature=0.0,
            max_tokens=100,
            prefer_local=True,
        )
        or ""
    )

    if isinstance(parsed, dict) and "value" in parsed:
        val = parsed.get("value")
        if val is None:
            return None
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return val
        if isinstance(val, str) and val.strip().isdigit():
            num = int(val.strip())
            return num if num > 0 else None

    if isinstance(raw_val, int):
        return raw_val if raw_val > 0 else None
    if raw_str.isdigit():
        num = int(raw_str)
        return num if num > 0 else None
    return default


def resolve_quiz_keyword_with_llm(
    raw_kw: Optional[str],
    user_text: str = "",
) -> Optional[str]:
    cleaned_kw = (raw_kw or "").strip()
    cleaned_user = (user_text or "").strip()
    if not cleaned_kw and not cleaned_user:
        return None

    prompt = (
        "당신은 뉴스 퀴즈 생성의 주제/키워드 필터를 판별하는 분류기입니다.\n"
        "후보 키워드와 사용자 원문을 보고 특정 주제/키워드로 좁혀야 하면 해당 키워드 문자열을, "
        "전체/랜덤/아무거나/일반 등 특정 주제 제한이 없으면 null을 반환하세요.\n\n"
        "규칙:\n"
        "1. '전체', '모두', '일반', '랜덤', '아무거나', '최근 뉴스'처럼 특정 주제/키워드가 아닌 경우 반드시 null을 출력하세요.\n"
        "2. '반도체', '금리', 'AI', '정치', '경제'처럼 특정 주제나 키워드가 명확하면 핵심 키워드만 문자열로 출력하세요.\n"
        "3. 마크다운, 코드블록, 설명 없이 오직 JSON 1개만 출력하세요.\n"
        '출력 형식: {"keyword": "반도체"} 또는 {"keyword": null}\n\n'
        f"후보 키워드: {cleaned_kw!r}\n"
        f"사용자 원문: {cleaned_user!r}"
    )

    parsed = extract_json(
        invoke_llm_with_fallback(
            messages=[HumanMessage(content=prompt)],
            temperature=0.0,
            max_tokens=100,
            prefer_local=True,
        )
        or ""
    )

    if isinstance(parsed, dict) and "keyword" in parsed:
        val = parsed.get("keyword")
        if val is None:
            return None
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ("null", "none"):
            return None
        return val_str

    return cleaned_kw or None


def extract_delete_args_with_llm(user_text: str, chosen: Dict[str, Any]) -> Dict[str, Any]:
    merged = unwrap_tool_args(chosen)
    schema = {
        "target_backend": "sqlite | postgres | null",
        "source": "언론사/섹션명만 (예: '연합', '조선일보', '동아/스포츠') | null",
        "category": "분야명 (예: '경제', '정치', 'IT', '과학', '스포츠') | null",
        "keyword": "본문/제목/키워드 검색어 | null",
        "days": "최근 N일 기준 정수 | null",
        "limit": "삭제할 건수 정수 | null",
        "order": "oldest | newest",
        "delete_all": "boolean (전체/전부/모두/다 삭제면 true, 조건/건수/키워드/언론사 삭제면 false)",
    }

    system_prompt = (
        "당신은 delete_news_records 도구의 인자를 사용자 요청에서 추출하는 JSON 추출기입니다.\n"
        "규칙:\n"
        "1. 마크다운, 코드블록, 설명 없이 순수 JSON 객체 1개만 출력하세요.\n"
        "2. 전체/전부/모두/다 삭제 요청이면 delete_all=true로 두고 나머지 필터(source, category, keyword, days, limit)는 null로 두세요.\n"
        "3. '오래된/옛날/먼저 들어온' 표현이 있으면 order='oldest', 그 외에는 'newest'로 설정하세요.\n"
        "4. 요청에 없는 값은 null로 채우세요.\n\n"
        f"추출 스키마:\n{json.dumps(schema, ensure_ascii=False)}"
    )

    raw = invoke_llm_with_fallback(
        messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text),
        ],
        temperature=0.0,
        max_tokens=300,
        prefer_local=True,
    )
    extracted = extract_json(raw or "") or {}

    result = dict(merged)
    for k, v in extracted.items():
        if v is not None:
            result[k] = v
    result["tool"] = "delete_news_records"
    return result