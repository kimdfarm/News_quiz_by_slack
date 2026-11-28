import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = os.getenv(
    "QUIZ_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# 지문 흐름(글 성격) 프로토타입 — 하드코딩된 키워드 대신 의미 문장으로 정의
PASSAGE_FLOW_PROTOTYPES: Dict[str, List[str]] = {
    "policy_regulation": [
        "정부가 관련 법령과 규제를 강화하고 위반 시 제재나 행정처분을 부과한다.",
        "제도 도입 배경과 정책 방향, 관리·감독 기준이 핵심 내용이다.",
    ],
    "tech_field_adoption": [
        "새로운 기술이나 시스템을 현장에 도입해 업무 효율과 처리 속도를 높인다.",
        "실제 적용 사례와 운영 성과, 현장 활용 과정이 중심이다.",
    ],
    "social_impact_esg": [
        "기업이나 기관이 재원 조성, 지원 사업, 상생 활동으로 사회적 가치를 만든다.",
        "기부나 협력 프로그램의 규모와 파급 효과가 주요 내용이다.",
    ],
    "debate_tradeoff": [
        "찬반 입장이나 장단점, 한계점이 맞서며 쟁점이 대립한다.",
        "기대 효과와 부작용 또는 리스크가 함께 논의된다.",
    ],
}

# 스레드 사용자 발화 의도 프로토타입
THREAD_INTENT_PROTOTYPES: Dict[str, List[str]] = {
    "rebuttal_or_challenge": [
        "제시된 정답이나 해설의 논리에 반박하며 다른 해석이나 이의를 제기한다.",
        "지문이나 기준에 비추어 판단이 타당한지 따져 묻고 반론을 펼친다.",
        "특정 선택지나 평가 결과가 왜 맞거나 틀린지 근거를 문제 삼는다.",
    ],
    "direct_answer_or_other": [
        "문제의 답을 선택하거나 제출한다.",
        "참고 자료나 채점 기준을 보여달라고 요청한다.",
    ],
}


@lru_cache(maxsize=2)
def _get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _embed_normalized(texts: List[str], model_name: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    model = _get_embedder(model_name)
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


@lru_cache(maxsize=8)
def _prototype_matrix(
    labels_tuple: Tuple[str, ...],
    protos_tuple: Tuple[Tuple[str, ...], ...],
    model_name: str,
) -> Tuple[List[str], np.ndarray]:
    """각 레이블별 프로토타입 임베딩 평균 벡터를 만듭니다."""
    label_names: List[str] = []
    vectors: List[np.ndarray] = []
    for label, samples in zip(labels_tuple, protos_tuple):
        if not samples:
            continue
        mat = _embed_normalized(list(samples), model_name=model_name)
        mean_vec = np.mean(mat, axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec = mean_vec / norm
        label_names.append(label)
        vectors.append(mean_vec.astype(np.float32))
    return label_names, np.vstack(vectors) if vectors else np.empty((0, 1), dtype=np.float32)


def classify_text_by_embedding(
    text: str,
    prototypes: Dict[str, List[str]],
    model_name: str = DEFAULT_EMBED_MODEL,
) -> Tuple[str, float, Dict[str, float]]:
    """입력 텍스트를 프로토타입 임베딩 코사인 유사도로 분류합니다."""
    cleaned = (text or "").strip()
    if not cleaned:
        first_label = next(iter(prototypes.keys()), "unknown")
        return first_label, 0.0, {}

    labels_tuple = tuple(prototypes.keys())
    protos_tuple = tuple(tuple(v) for v in prototypes.values())
    label_names, proto_mat = _prototype_matrix(labels_tuple, protos_tuple, model_name)
    if proto_mat.size == 0:
        return "unknown", 0.0, {}

    query_vec = _embed_normalized([cleaned], model_name=model_name)[0]
    scores_arr = proto_mat @ query_vec  # 정규화 벡터 내적 == 코사인 유사도
    scores = {lbl: float(scores_arr[i]) for i, lbl in enumerate(label_names)}
    best_idx = int(np.argmax(scores_arr))
    return label_names[best_idx], float(scores_arr[best_idx]), scores


def classify_articles_flow(
    articles: List[Dict[str, Any]],
    model_name: str = DEFAULT_EMBED_MODEL,
) -> Dict[str, Any]:
    """기사들을 Sentence 임베딩으로 분류해 글 전개 순서와 대표 성격을 반환합니다."""
    classified_items: List[Dict[str, Any]] = []
    for art in articles or []:
        text = " ".join(
            str(art.get(k) or "").strip()
            for k in ("title", "facts", "summary", "content", "opinion")
            if art.get(k)
        )
        flow_label, score, all_scores = classify_text_by_embedding(
            text, PASSAGE_FLOW_PROTOTYPES, model_name=model_name
        )
        classified_items.append(
            {
                "article": art,
                "flow_type": flow_label,
                "flow_score": round(score, 4),
                "scores": {k: round(v, 4) for k, v in all_scores.items()},
            }
        )

    # 지문 전개 자연스러움을 위한 기본 순서: 제도/배경 -> 현장 적용/사례 -> 쟁점/대조 -> 사회/기타 영향
    flow_order = {
        "policy_regulation": 0,
        "tech_field_adoption": 1,
        "debate_tradeoff": 2,
        "social_impact_esg": 3,
    }
    ordered = sorted(
        classified_items,
        key=lambda item: (flow_order.get(item["flow_type"], 99), -item["flow_score"]),
    )
    ordered_articles = [item["article"] for item in ordered]

    counts: Dict[str, int] = {}
    for item in ordered:
        counts[item["flow_type"]] = counts.get(item["flow_type"], 0) + 1

    dominant_flows = [
        k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    flow_outline = " -> ".join(dominant_flows) if dominant_flows else "general"

    return {
        "ordered_articles": ordered_articles,
        "flow_outline": flow_outline,
        "items": ordered,
    }


def is_rebuttal_semantic(
    user_text: str,
    threshold: float = 0.42,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> Tuple[bool, str, float]:
    """사용자 스레드 메시지가 반박/이의제기 성격인지 임베딩 유사도로 판별합니다."""
    label, score, _ = classify_text_by_embedding(
        user_text, THREAD_INTENT_PROTOTYPES, model_name=model_name
    )
    return (label == "rebuttal_or_challenge" and score >= threshold), label, score