import json
import os
import re
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
STATE_FILE_PATH = os.path.join(PROJECT_ROOT, "storage_mode_state.json")

VALID_MODES = ("postgres", "sqlite")
NEWS_KEYWORDS = (
    "뉴스", "기사", "수집", "가져와", "모아줘", "크롤", "rss",
    "연합", "경향", "조선", "동아", "한경", "매경", "한겨레", "sbs", "뉴시스",
)


def normalize_storage_mode(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    val = str(raw).strip().strip('"').strip("'").lower()
    if val in ("sqlite", "file"):
        return "sqlite"
    if val in ("postgres", "postgresql", "db", "디비", "데이터베이스"):
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


def parse_storage_mode_command(request_text: str) -> Optional[str]:
    """
    요청 문장이 저장소 모드 전환 명령이면 'sqlite' 또는 'postgres'를 반환하고,
    일반 수집 요청이면 None을 반환합니다.
    """
    text = (request_text or "").strip()
    if not text:
        return None

    change_keywords = ("바꿔", "변경", "전환", "설정", "고정", "모드로", "방식으로", "앞으로", "계속", "기본값")
    has_change_intent = any(k in text for k in change_keywords)

    if re.search(r"(sqlite)\s*(로|으로)?\s*(바꿔|변경|전환|설정|고정|저장\s*모드|해줘|해\s*줘)?", text, re.IGNORECASE) and has_change_intent:
        return "sqlite"

    if re.search(r"(db|디비|postgres|postgresql|데이터베이스)\s*(로|으로)?\s*(바꿔|변경|전환|설정|고정|저장\s*모드|해줘|해\s*줘)?", text, re.IGNORECASE) and has_change_intent:
        return "postgres"

    if not any(nk in text.lower() for nk in NEWS_KEYWORDS):
        if re.search(r"\bsqlite\b", text, re.IGNORECASE):
            return "sqlite"
        if re.search(r"\b(postgres|postgresql|db|디비)\b", text, re.IGNORECASE):
            return "postgres"

    return None


def is_mode_only_request(request_text: str) -> bool:
    """뉴스 수집 의도 없이 저장 모드 변경만 요청한 경우인지 판별합니다."""
    text = (request_text or "").strip()
    if not text or not parse_storage_mode_command(text):
        return False
    return not any(nk in text.lower() for nk in NEWS_KEYWORDS)


def resolve_storage_mode_for_request(
    request_text: str = "",
    explicit_storage_mode: Optional[str] = None,
    persist_command: bool = True,
) -> str:
    """
    요청별 실제 적용할 storage_mode를 결정합니다.
    - explicit_storage_mode가 주어지면 그 값을 사용('auto'면 저장된 모드 사용)
    - 요청에 모드 전환 명령이 있으면 persist_command=True일 때 서버 상태 파일에 저장 후 반환
    - 그 외에는 서버에 저장된 모드를 반환
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