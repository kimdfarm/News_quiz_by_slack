import json
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
STATE_FILE_PATH = os.path.join(PROJECT_ROOT, "storage_mode_state.json")

VALID_MODES = ("postgres", "sqlite")


def normalize_storage_mode(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    val = str(raw).strip().strip('"').strip("'").lower()
    if val in ("sqlite", "file", "local_file", "sq", "로컬", "로컬파일", "파일"):
        return "sqlite"
    if val in ("postgres", "postgresql", "db", "pg", "neon", "neondb", "네온", "네온db", "디비", "데이터베이스"):
        return "postgres"
    return None


def get_persisted_storage_mode() -> str:
    """
    저장 모드 우선순위:
    1) storage_mode_state.json에 저장된 값
    2) 환경변수 STORAGE_BACKEND 초기값
    3) 기본값 'postgres'
    """
    if os.path.exists(STATE_FILE_PATH):
        try:
            with open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                mode = normalize_storage_mode(data.get("storage_mode"))
                if mode in VALID_MODES:
                    return mode
        except Exception:
            pass

    env_mode = normalize_storage_mode(os.getenv("STORAGE_BACKEND"))
    if env_mode in VALID_MODES:
        return env_mode

    return "postgres"


def set_persisted_storage_mode(mode: str) -> str:
    """서버 상태 파일(storage_mode_state.json)에 저장 모드를 기록합니다."""
    normalized = normalize_storage_mode(mode)
    if normalized not in VALID_MODES:
        raise ValueError(f"지원하지 않는 storage_mode입니다: {mode}")

    os.makedirs(os.path.dirname(STATE_FILE_PATH), exist_ok=True)
    with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump({"storage_mode": normalized}, f, ensure_ascii=False, indent=2)
    return normalized


def _get_groq_models() -> list[str]:
    raw_list = (os.getenv("GROQ_MODELS") or "").strip()
    if raw_list:
        models = [m.strip() for m in raw_list.split(",") if m.strip()]
        if models:
            return list(dict.fromkeys(models))

    primary = (os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b").strip()
    fallbacks_raw = (os.getenv("GROQ_FALLBACK_MODELS") or "openai/gpt-oss-20b,qwen/qwen3.8-27b").strip()
    fallbacks = [m.strip() for m in fallbacks_raw.split(",") if m.strip()]
    return list(dict.fromkeys([primary] + fallbacks))


def _is_ollama_model_available(model_name: str, base_url: str, timeout: float = 1.5) -> bool:
    """로컬 Ollama 서버에 모델이 실제로 존재하는지 확인합니다. 없으면 가동하지 않습니다."""
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


def _extract_json_dict(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _infer_storage_intent_with_llm(request_text: str) -> Optional[Dict[str, Any]]:
    """
    요청에서 저장소 전환 여부/대상과 '모드 변경 전용 요청인지'를 LLM으로 판별합니다.
    반환 예시: {"target_mode": "sqlite" | "postgres" | null, "mode_only": true | false}
    """
    text = (request_text or "").strip()
    if not text:
        return None

    system_prompt = (
        "당신은 사용자의 자연어 요청에서 '저장소 백엔드 변경 의도'를 판별하는 분류기입니다.\n"
        "유효한 JSON만 출력하세요.\n"
        "출력 스키마:\n"
        '{"target_mode": "sqlite" | "postgres" | null, "mode_only": true | false}\n\n'
        "판별 기준:\n"
        "1. 사용자가 저장 위치/방식/DB를 바꾸려 하면 target_mode를 정하세요.\n"
        "   - sqlite, 로컬 파일, 파일 저장 등 -> \"sqlite\"\n"
        "   - postgres, postgresql, neon, 네온 DB, db, 디비, 데이터베이스 등 -> \"postgres\"\n"
        "   - 저장소 변경 의도가 전혀 없으면 -> null\n"
        "2. mode_only:\n"
        "   - 저장소 변경만 요청하고 뉴스 수집/크롤링/퀴즈 조회 등 다른 작업 요청이 없으면 true\n"
        "   - 저장소 변경과 뉴스 수집·조회 요청이 함께 있거나 저장소 변경 요청이 아니면 false"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"사용자 요청: {text}"),
    ]

    groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if groq_api_key:
        for model_name in _get_groq_models():
            try:
                llm = ChatGroq(
                    model=model_name,
                    api_key=groq_api_key,
                    temperature=0.0,
                    max_tokens=200,
                )
                resp = llm.invoke(messages)
                raw = resp.content if isinstance(resp.content, str) else str(resp.content)
                parsed = _extract_json_dict(raw)
                if parsed is not None:
                    return parsed
            except Exception as e:
                print(f"[!] storage_mode_config Groq({model_name}) 실패: {e}", file=sys.stderr)

    ollama_model = (os.getenv("OLLAMA_MODEL") or "gemma3:4b").strip()
    ollama_base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
    if not _is_ollama_model_available(ollama_model, ollama_base_url):
        print(
            f"[i] 로컬 Ollama에 '{ollama_model}' 모델이 없어 가동하지 않습니다.",
            file=sys.stderr,
        )
        return None

    try:
        ollama_llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.0,
        )
        resp = ollama_llm.invoke(messages)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        return _extract_json_dict(raw)
    except Exception as e:
        print(f"[!] storage_mode_config Ollama({ollama_model}) 실패: {e}", file=sys.stderr)
        return None


def analyze_storage_request(request_text: str) -> Tuple[Optional[str], bool]:
    """
    (target_mode, mode_only)를 반환합니다.
    target_mode가 None이면 모드 변경 요청이 아닙니다.
    """
    inferred = _infer_storage_intent_with_llm(request_text)
    if not inferred:
        return None, False

    raw_target = inferred.get("target_mode")
    if raw_target is None or str(raw_target).strip().lower() in ("null", "none", ""):
        return None, False

    normalized = normalize_storage_mode(str(raw_target))
    if normalized not in VALID_MODES:
        return None, False

    mode_only = bool(inferred.get("mode_only"))
    return normalized, mode_only


def parse_storage_mode_command(request_text: str) -> Optional[str]:
    """요청 문장이 저장소 모드 전환을 포함하면 'sqlite' 또는 'postgres'를 반환합니다."""
    mode, _ = analyze_storage_request(request_text)
    return mode


def is_mode_only_request(request_text: str) -> bool:
    """다른 작업 없이 저장 모드 변경만 요청한 경우인지 판별합니다."""
    mode, mode_only = analyze_storage_request(request_text)
    return mode in VALID_MODES and mode_only


def resolve_storage_mode_for_request(
    request_text: str = "",
    explicit_storage_mode: Optional[str] = None,
    persist_command: bool = True,
) -> str:
    """
    요청별 실제 적용할 storage_mode를 결정합니다.
    - explicit_storage_mode가 주어지면 그 값을 사용
    - 요청에 모드 전환 의도가 있으면 persist_command=True일 때 storage_mode_state.json에 저장 후 반환
    - 그 외에는 저장된 모드를 반환
    """
    explicit = normalize_storage_mode(explicit_storage_mode)
    if explicit in VALID_MODES:
        return explicit

    commanded = parse_storage_mode_command(request_text)
    if commanded in VALID_MODES:
        if persist_command:
            return set_persisted_storage_mode(commanded)
        return commanded

    return get_persisted_storage_mode()