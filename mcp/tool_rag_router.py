import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# 동의어 및 활용형 정규화 매핑
SYNONYM_MAP: Dict[str, str] = {
    "바꿔": "변경",
    "바꾸": "변경",
    "전환": "변경",
    "교체": "변경",
    "내보내": "추출",
    "내보내기": "추출",
    "다운": "다운로드",
    "받아": "다운로드",
    "보내": "추출",
    "지워": "삭제",
    "비워": "삭제",
    "없애": "삭제",
    "제거": "삭제",
    "모아": "수집",
    "긁어": "수집",
    "가져와": "수집",
    "크롤링": "수집",
    "네온": "postgres",
    "디지털": "postgres",
    "postgresql": "postgres",
    "로컬": "sqlite",
}

MCP_TOOL_REGISTRY: List[Dict[str, Any]] = [
    {
        "tool": "delete_news_records",
        "group": "storage",
        "intent_tags": ["삭제", "지워", "비워", "없애", "제거"],
        "tags": ["오래된", "최근", "전체", "전부", "언론사", "키워드", "기간", "건수", "db", "sqlite", "postgres"],
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
            "source": "언론사/섹션명만 | null",
            "category": "분야명 | null",
            "keyword": "키워드 | null",
            "days": "정수 | null",
            "limit": "정수 | null",
            "order": "oldest | newest",
            "delete_all": "boolean (전부/모두/전체 삭제면 true, 특정 조건 삭제면 false)",
        },
    },
    {
        "tool": "get_storage_stats",
        "group": "storage",
        "intent_tags": ["통계", "건수", "몇 건", "분포", "현황"],
        "tags": ["언론사별", "분야별", "기간", "db", "sqlite", "postgres"],
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
        "intent_tags": ["상태", "연결", "확인", "어디"],
        "tags": ["모드", "저장소", "db", "sqlite", "postgres", "네온"],
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
        "intent_tags": ["변경", "바꿔", "전환", "모드"],
        "tags": ["저장소", "형태", "sqlite", "파일", "로컬", "postgres", "네온", "db"],
        "description": "활성 저장소 백엔드(저장 모드)를 sqlite(로컬 파일 형태) 또는 postgres(Neon DB)로 전환한다. 파일을 다운로드하는 도구가 아니다.",
        "examples": [
            "네온db로 변경해줘",
            "sqlite로 바꿔줘",
            "저장 상태를 파일 형태로 바꿔줘",
            "파일형태를 추출할 수 있는 sqlite형태로 변경",
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
        "intent_tags": ["추출", "다운로드", "내보내기", "보내줘"],
        "tags": ["파일", "sqlite", "db"],
        "description": "로컬 SQLite DB 파일(news_store.sqlite) 자체를 Slack 채널로 업로드해 다운로드하게 한다. 저장소 모드를 변경하는 도구가 아니다.",
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
        "intent_tags": ["동기화", "올려", "이동"],
        "tags": ["sqlite", "파일", "로컬", "postgres", "네온", "디지털"],
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
        "intent_tags": ["동기화", "내려받기", "이동"],
        "tags": ["postgres", "네온", "디지털", "sqlite", "파일", "로컬"],
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
        "intent_tags": ["수집", "크롤링", "모아줘", "가져와", "긁어와"],
        "tags": ["저장", "뉴스", "기사", "정보", "rss", "언론사", "분야"],
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
        "intent_tags": ["퀴즈", "문제", "출제", "내줘"],
        "tags": ["시사", "뉴스", "만들어줘"],
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
    {
        "tool": "schedule_rss_collection",
        "group": "content",
        "intent_tags": ["주기", "마다", "간격", "동안", "까지", "예약", "자동"],
        "tags": [
            "시간", "분", "초", "요일", "주", "내일", "모레", "년", "월", "일",
            "수집", "크롤링", "모아줘", "가져와", "긁어와", "뉴스", "기사", "rss",
        ],
        "description": (
            "현재 시각 기준으로 지정한 기간(시/분/초 동안) 또는 목표 시각/날짜/요일(L시까지, 내일/금요일/연월일)까지 "
            "지정한 간격(시/분/초마다, 기본 30분)으로 RSS 뉴스를 반복 수집한다."
        ),
        "examples": [
            "2시간 30분 15초 동안 15분마다 경제 뉴스 3개씩 모아줘",
            "오늘 밤 11시 30분까지 10분마다 정치 기사 2개씩 수집해줘",
            "다음주 금요일 18시까지 1시간 30분마다 IT 뉴스 가져와줘",
            "2026년 11월 25일 09시 00분까지 30분마다 뉴스 수집해줘",
        ],
        "schema": {
            "tool": "schedule_rss_collection",
            "mode": "duration | until",
            "duration_hours": "실수 또는 정수 (기본 0)",
            "duration_minutes": "실수 또는 정수 (기본 0)",
            "duration_seconds": "실수 또는 정수 (기본 0)",
            "until_iso": "YYYY-MM-DD HH:MM:SS 형식 목표 시각(until 모드일 때) 또는 null",
            "interval_hours": "실수 또는 정수 (기본 0)",
            "interval_minutes": "실수 또는 정수 (기본 30, 다른 간격 지정 시 0 가능)",
            "interval_seconds": "실수 또는 정수 (기본 0)",
            "request_text": "수집 대상/건수 요청 문장(예: '경제 뉴스 3개씩 수집')",
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

        # 복합어 내부 핵심 영문/한글 키워드 분해 (예: 네온db -> 네온, db / 파일형태 -> 파일, 형태)
        sub_parts = re.findall(r"[a-z]+|[0-9]+|[가-힣]+", raw)
        if len(sub_parts) > 1:
            tokens.extend(sub_parts)

        # 한국어 조사 및 동사/형용사 활용 어미 제거
        trimmed = re.sub(
            r"(을|를|이|가|은|는|에|에서|로|으로|와|과|의|도|만|까지|부터|"
            r"해줘|해주세요|해봐|해|줘|할|하는|한|된|될|되는|해서|하여|하고|싶어)$",
            "",
            raw,
        )
        if trimmed and trimmed != raw:
            tokens.append(trimmed)

    # 동의어/어간 표준화 추가
    normalized: List[str] = []
    for tok in tokens:
        normalized.append(tok)
        if tok in SYNONYM_MAP:
            normalized.append(SYNONYM_MAP[tok])

    return list(dict.fromkeys(normalized))


def _apply_intent_rules(tool_name: str, lowered_text: str, tokens: Set[str]) -> float:
    """문장 끝/주서술어 중심의 의도 보정 규칙."""
    bonus = 0.0

    # 저장소 백엔드 전환 의도 ("~로 변경", "~로 바꿔", "~형태로 전환")
    is_backend_switch = bool(
        re.search(r"(?:형태|모드|방식|저장소|sqlite|postgres|네온|db)\s*(?:으?로)?\s*(?:변경|바꿔|바꾸|전환)", lowered_text)
    )
    if is_backend_switch:
        if tool_name == "set_storage_backend":
            bonus += 5.0
        elif tool_name == "export_sqlite_file":
            # '추출할 수 있는 sqlite형태로 변경'처럼 수식절에 '추출'이 쓰인 경우 감점
            bonus -= 3.5

    # 순수 파일 추출/다운로드 의도 ("파일 추출", "db 다운로드", "sqlite 보내줘")
    is_file_export = bool(
        re.search(r"(?:파일|sqlite|db)\s*(?:을|를)?\s*(?:추출|다운로드|내보내|보내)", lowered_text)
    ) and not is_backend_switch
    if is_file_export and tool_name == "export_sqlite_file":
        bonus += 4.5

    # 동기화 방향 명확화
    if "동기화" in tokens or "이동" in tokens:
        if re.search(r"(?:sqlite|파일|로컬).*(?:postgres|네온|디지털)", lowered_text) and tool_name == "sync_sqlite_to_postgres":
            bonus += 4.0
        elif re.search(r"(?:postgres|네온|디지털).*(?:sqlite|파일|로컬)", lowered_text) and tool_name == "sync_postgres_to_sqlite":
            bonus += 4.0

    return bonus


def _get_optional_embedding(text: str) -> Optional[List[float]]:
    cache_key = " ".join((text or "").lower().split())
    if cache_key in _EMBED_CACHE:
        return _EMBED_CACHE[cache_key]

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
                    _EMBED_CACHE[cache_key] = vec
                    return vec
        except Exception:
            pass

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
                    _EMBED_CACHE[cache_key] = vec
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

        # 1) 핵심 의도 태그 (높은 가중치)
        intent_tags = [str(t).lower() for t in (item.get("intent_tags") or [])]
        for tag in intent_tags:
            norm_tag = SYNONYM_MAP.get(tag, tag)
            if tag in query_tokens or norm_tag in query_tokens or tag in lowered_text:
                kw_score += 4.5

        # 2) 보조 대상 명사 태그 (낮은 가중치: 공통 명사 쏠림 방지)
        noun_tags = [str(t).lower() for t in (item.get("tags") or [])]
        for tag in noun_tags:
            norm_tag = SYNONYM_MAP.get(tag, tag)
            if tag in query_tokens or norm_tag in query_tokens:
                kw_score += 1.2

        # 3) 그룹 힌트 점수
        grp = item.get("group", "")
        if grp in group_scores and group_scores[grp] > 0:
            kw_score += min(group_scores[grp], 3) * 0.8

        # 4) 설명 및 예시 토큰 겹침
        doc_blob = f"{item.get('description', '')} {' '.join(item.get('examples') or [])}"
        doc_tokens = set(_normalize_tokens(doc_blob))
        kw_score += len(query_tokens & doc_tokens) * 0.7

        # 5) 도구명 직접 매칭
        tool_parts = set(re.split(r"[_\s]+", str(item.get("tool", "")).lower()))
        kw_score += len(query_tokens & tool_parts) * 1.5

        # 6) 의도 충돌/보강 규칙 적용
        kw_score += _apply_intent_rules(item.get("tool", ""), lowered_text, query_tokens)

        total_score = kw_score
        if query_vec is not None:
            doc_vec = _get_optional_embedding(doc_blob)
            if doc_vec is not None:
                sem_score = max(0.0, _cosine_similarity(query_vec, doc_vec))
                total_score = kw_score + (sem_score * 5.0)

        scored.append((total_score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    if min_score > 0:
        selected = [item for score, item in scored if score >= min_score][:top_k]
    else:
        selected = [item for _, item in scored[:top_k]]

    return ensure_rss_pair_in_candidates(
        top_tools=selected,
        registry=MCP_TOOL_REGISTRY,
        max_k=max(top_k, 4),
    )


def ensure_rss_pair_in_candidates(
    top_tools: List[Dict[str, Any]],
    registry: List[Dict[str, Any]],
    max_k: int = 5,
) -> List[Dict[str, Any]]:
    by_name = {item["tool"]: item for item in registry}
    present = {item["tool"] for item in top_tools}

    has_one_shot = "run_rss_by_request" in present
    has_sched = "schedule_rss_collection" in present

    if not (has_one_shot or has_sched):
        return top_tools

    result = list(top_tools)
    for peer_name in ("run_rss_by_request", "schedule_rss_collection"):
        if peer_name not in present and peer_name in by_name:
            result.append(by_name[peer_name])
            present.add(peer_name)

    if len(result) > max_k:
        protected = {"run_rss_by_request", "schedule_rss_collection"}
        trimmed: List[Dict[str, Any]] = []
        for item in result:
            if item["tool"] in protected or len(trimmed) < max_k - len(protected):
                trimmed.append(item)
        existing_names = {x["tool"] for x in trimmed}
        for item in result:
            if item["tool"] in protected and item["tool"] not in existing_names:
                trimmed.append(item)
        result = trimmed

    return result