import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

MCP_TOOL_REGISTRY: List[Dict[str, Any]] = [
    {
        "tool": "delete_news_records",
        "group": "storage",
        "tags": ["삭제", "지워", "비워", "없애", "오래된", "최근", "전체", "전부", "언론사", "키워드", "기간", "건수"],
        "description": "언론사/섹션, 분야, 키워드, 최근 일수(days), 건수(limit), 정렬순서(oldest/newest) 또는 전체(delete_all) 조건으로 저장된 뉴스 데이터를 삭제한다.",
        "examples": [
            "오래된 연합/정치 정보 3개 삭제해줘",
            "조선일보 정보 전부 삭제해줘",
            "동아/스포츠 기사 지워줘",
            "최근 2개 삭제해줘",
            "오래된 데이터 5개 지워줘",
            "환율 들어간 기사 삭제해줘",
            "최근 3일 뉴스 지워줘",
            "sqlite 데이터 다 지워줘",
        ],
        "schema": {
            "tool": "delete_news_records",
            "target_backend": "sqlite | postgres | null",
            "source": "언론사/섹션명만('정보','기사','데이터' 제외) | null",
            "category": "분야명 | null",
            "keyword": "키워드 | null",
            "days": "정수 | null",
            "limit": "정수 | null",
            "order": "oldest | newest",
            "delete_all": "boolean (다른 필터가 있으면 반드시 false)",
        },
    },
    {
        "tool": "get_storage_stats",
        "group": "storage",
        "tags": ["통계", "건수", "몇 건", "분포", "언론사별", "분야별", "기간", "현황", "db", "sqlite", "postgres"],
        "description": "저장소(sqlite 또는 postgres)의 총 기사 수, 언론사별·분야별 분포, 기사 기간 통계를 조회한다.",
        "examples": [
            "현재 db 통계",
            "저장된 정보 통계 보여줘",
            "몇 건 쌓였어?",
            "언론사별 기사 수 알려줘",
            "sqlite 통계 확인해줘",
        ],
        "schema": {
            "tool": "get_storage_stats",
            "target_backend": "sqlite | postgres | null",
        },
    },
    {
        "tool": "get_storage_status",
        "group": "storage",
        "tags": ["상태", "연결", "확인", "모드", "저장소", "어디", "db", "sqlite", "postgres", "네온"],
        "description": "현재 활성 저장소 모드와 SQLite 파일 및 PostgreSQL 연결 상태를 확인한다.",
        "examples": [
            "저장소 상태 확인해줘",
            "DB 연결 상태 보고해줘",
            "현재 어디에 저장 중이야?",
            "현재 db 상태 어때?",
        ],
        "schema": {"tool": "get_storage_status"},
    },
    {
        "tool": "set_storage_backend",
        "group": "storage",
        "tags": ["변경", "바꿔", "전환", "모드", "저장소", "sqlite", "파일", "로컬", "postgres", "네온", "db"],
        "description": "활성 저장소 모드를 sqlite(로컬 파일) 또는 postgres(Neon DB)로 전환한다.",
        "examples": [
            "네온db로 변경해줘",
            "sqlite로 바꿔줘",
            "저장 상태를 파일 형태로 바꿔줘",
            "디지털 DB로 저장하게 해줘",
        ],
        "schema": {
            "tool": "set_storage_backend",
            "storage_backend": "sqlite | postgres",
        },
    },
    {
        "tool": "export_sqlite_file",
        "group": "storage",
        "tags": ["추출", "다운로드", "내보내기", "보내줘", "파일", "sqlite", "db"],
        "description": "로컬 SQLite DB 파일(news_store.sqlite)을 Slack으로 추출/다운로드한다.",
        "examples": [
            "파일 추출해",
            "sqlite 파일 추출해줘",
            "news_store.sqlite 다운로드",
            "db 파일 보내줘",
        ],
        "schema": {"tool": "export_sqlite_file"},
    },
    {
        "tool": "sync_sqlite_to_postgres",
        "group": "storage",
        "tags": ["동기화", "올려", "이동", "가져와", "sqlite", "파일", "로컬", "postgres", "네온", "디지털"],
        "description": "로컬 SQLite 파일 데이터를 Neon DB(PostgreSQL)로 단방향 동기화한다.",
        "examples": [
            "sqlite를 postgres로 동기화해줘",
            "로컬 데이터 네온으로 올려줘",
            "파일 데이터를 디지털 db로 이동해줘",
        ],
        "schema": {
            "tool": "sync_sqlite_to_postgres",
            "limit": "정수 | null",
        },
    },
    {
        "tool": "sync_postgres_to_sqlite",
        "group": "storage",
        "tags": ["동기화", "내려받기", "가져와", "이동", "postgres", "네온", "디지털", "sqlite", "파일", "로컬"],
        "description": "Neon DB(PostgreSQL) 데이터를 로컬 SQLite 파일로 단방향 내려받기 동기화한다.",
        "examples": [
            "postgres를 sqlite로 동기화해줘",
            "네온 db 데이터를 로컬 파일로 가져와줘",
            "디지털 db 데이터를 파일로 이동해줘",
        ],
        "schema": {
            "tool": "sync_postgres_to_sqlite",
            "limit": "정수 | null",
        },
    },
    {
        "tool": "run_rss_by_request",
        "group": "content",
        "tags": ["수집", "크롤링", "모아줘", "가져와", "긁어와", "저장", "뉴스", "기사", "정보", "rss", "언론사", "분야"],
        "description": "자연어 요청을 해석해 언론사/분야별 RSS 뉴스를 크롤링 및 수집한다.",
        "examples": [
            "언론사별 각 정보 4개씩 가져와",
            "국제 뉴스 3개 모아줘",
            "연합/경제 수집해줘",
            "정치 기사 5개 긁어와",
        ],
        "schema": {"tool": "run_rss_by_request"},
    },
    {
        "tool": "get_recent_news_for_quiz",
        "group": "content",
        "tags": ["퀴즈", "문제", "출제", "만들어줘", "내줘", "시사", "뉴스"],
        "description": "저장된 최근 뉴스를 조회해 시사 퀴즈 문제를 생성한다.",
        "examples": [
            "경제 뉴스로 퀴즈 내줘",
            "최근 뉴스 문제 만들어줘",
            "시사 퀴즈 1문제 출제해줘",
        ],
        "schema": {
            "tool": "get_recent_news_for_quiz",
            "category": "분야명 또는 전체",
            "limit": "정수 (기본 3)",
        },
    },
]

GROUP_HINTS: Dict[str, Set[str]] = {
    "storage": {
        "db", "sqlite", "postgres", "postgresql", "네온", "파일", "로컬",
        "디지털", "저장소", "동기화", "통계", "상태", "추출", "삭제", "지워", "비워",
    },
    "content": {
        "뉴스", "기사", "정보", "수집", "크롤링", "rss", "퀴즈", "문제", "출제", "언론사", "분야",
    },
}

_EMBED_CACHE: Dict[str, List[float]] = {}


def _normalize_tokens(text: str) -> List[str]:
    lowered = (text or "").lower()
    cleaned = re.sub(r"[^0-9a-zA-Z가-힣/]+", " ", lowered)
    tokens: List[str] = []
    for raw in cleaned.split():
        tokens.append(raw)
        trimmed = re.sub(
            r"(을|를|이|가|은|는|에|에서|로|으로|와|과|의|도|만|까지|부터|해줘|해주세요|해봐|해|줘)$",
            "",
            raw,
        )
        if trimmed and trimmed != raw:
            tokens.append(trimmed)
    return tokens


def _get_optional_embedding(text: str) -> Optional[List[float]]:
    """임베딩 설정이 있을 때만 벡터를 구하고, 없으면 None을 반환."""
    if text in _EMBED_CACHE:
        return _EMBED_CACHE[text]

    # 1) 로컬 Ollama 임베딩 모델
    ollama_embed_model = (os.getenv("OLLAMA_EMBED_MODEL") or "").strip()
    if ollama_embed_model:
        base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
        try:
            resp = requests.post(
                f"{base_url.rstrip('/')}/api/embeddings",
                json={"model": ollama_embed_model, "prompt": text},
                timeout=1.5,
            )
            if resp.status_code == 200:
                vec = (resp.json() or {}).get("embedding")
                if isinstance(vec, list) and vec:
                    _EMBED_CACHE[text] = vec
                    return vec
        except Exception:
            pass

    # 2) 상용 OpenAI 호환 임베딩 엔드포인트(설정된 경우에만)
    openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    openai_embed_model = (os.getenv("OPENAI_EMBED_MODEL") or "").strip()
    if openai_api_key and openai_embed_model:
        base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
        try:
            resp = requests.post(
                f"{base_url.rstrip('/')}/embeddings",
                headers={
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": openai_embed_model, "input": text},
                timeout=2.0,
            )
            if resp.status_code == 200:
                data = (resp.json() or {}).get("data") or []
                vec = (data[0] or {}).get("embedding") if data else None
                if isinstance(vec, list) and vec:
                    _EMBED_CACHE[text] = vec
                    return vec
        except Exception:
            pass

    return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve_top_tools(
    user_text: str,
    top_k: int = 3,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    임베딩이 있으면 의미 유사도 + 키워드/태그/그룹 점수로,
    임베딩이 없으면 키워드/태그/그룹 점수로 상위 top_k 도구를 추립니다.
    """
    lowered_text = (user_text or "").lower()
    query_tokens = set(_normalize_tokens(user_text))
    query_vec = _get_optional_embedding(user_text)

    group_scores: Dict[str, int] = {g: 0 for g in GROUP_HINTS}
    for g, hints in GROUP_HINTS.items():
        for h in hints:
            if h in lowered_text or h in query_tokens:
                group_scores[g] += 1

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in MCP_TOOL_REGISTRY:
        kw_score = 0.0

        tags = [str(t).lower() for t in (item.get("tags") or [])]
        for tag in tags:
            if tag in lowered_text or tag in query_tokens:
                kw_score += 3.0

        grp = item.get("group", "")
        if grp in group_scores and group_scores[grp] > 0:
            kw_score += min(group_scores[grp], 3) * 1.0

        doc_blob = f"{item.get('description', '')} {' '.join(item.get('examples') or [])}"
        doc_tokens = set(_normalize_tokens(doc_blob))
        kw_score += len(query_tokens & doc_tokens) * 0.8

        tool_parts = set(re.split(r"[_\s]+", str(item.get("tool", "")).lower()))
        kw_score += len(query_tokens & tool_parts) * 1.5

        total_score = kw_score
        if query_vec is not None:
            doc_vec = _get_optional_embedding(doc_blob)
            if doc_vec is not None:
                sem_score = max(0.0, _cosine_similarity(query_vec, doc_vec))
                total_score = kw_score + (sem_score * 5.0)

        scored.append((total_score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    if min_score > 0:
        filtered = [item for score, item in scored if score >= min_score]
        return filtered[:top_k]
    return [item for _, item in scored[:top_k]]