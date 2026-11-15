import os
import re
import math
import hashlib
from collections import Counter
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

CONTENT_SIMILARITY_THRESHOLD =0.88


def get_db_connection():
    """PostgreSQL 데이터베이스 연결을 생성하고 반환합니다."""
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def init_db():
    """
    이슈 중심 정규화 스키마 초기화:
    1. categories, sources (마스터 테이블 - 공간 절약)
    2. news_issues (기사/이슈 내용 마스터)
    3. news_perspectives (기자/언론사별 의견 및 연계 상세)
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"pgvector 확장 로드 스킵 (기본 해시/비교기 매칭으로 동작): {e}")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sources (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS news_issues (
            id SERIAL PRIMARY KEY,
            issue_title TEXT NOT NULL,
            representative_summary TEXT,
            category_id INT REFERENCES categories(id) ON DELETE SET NULL,
            keywords TEXT[],
            issue_hash VARCHAR(64) UNIQUE,
            published_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS news_perspectives (
            id SERIAL PRIMARY KEY,
            issue_id INT REFERENCES news_issues(id) ON DELETE CASCADE,
            source_id INT REFERENCES sources(id) ON DELETE SET NULL,

            title TEXT NOT NULL,
            author_name TEXT,
            url TEXT UNIQUE NOT NULL,

            summary_facts TEXT,
            author_opinion TEXT,
            stance_type VARCHAR(20) DEFAULT '중립',

            raw_content TEXT,
            content_hash VARCHAR(64),
            thumbnail_url TEXT,

            processing_status VARCHAR(20) DEFAULT 'COMPLETED',
            model_version VARCHAR(50),
            metadata JSONB,

            published_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_issues_published_at ON news_issues(published_at DESC);
        CREATE INDEX IF NOT EXISTS idx_issues_category_id ON news_issues(category_id);
        CREATE INDEX IF NOT EXISTS idx_perspectives_issue_id ON news_perspectives(issue_id);
        CREATE INDEX IF NOT EXISTS idx_perspectives_source_id ON news_perspectives(source_id);
        CREATE INDEX IF NOT EXISTS idx_perspectives_content_hash ON news_perspectives(content_hash);
        CREATE INDEX IF NOT EXISTS idx_perspectives_published_at ON news_perspectives(published_at DESC);
    """)
    conn.commit()
    cur.close()
    conn.close()


def normalize_text_for_comparison(text: str) -> str:
    """비교기(해시/유사도)용 텍스트 정규화: 공백·기호·메타 문구 정리."""
    if not text:
        return ""
    cleaned = re.sub(r"\[RSS 요약\]|\[본문\]|\[\.\.\.중략\.\.\.\]", " ", text)
    cleaned = re.sub(r"[^\w\s가-힣a-zA-Z0-9]", " ", cleaned.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def generate_content_hash(content: str) -> str:
    """정규화된 기사 원문 텍스트 기반으로 SHA-256 해시값을 생성합니다."""
    norm = normalize_text_for_comparison(content) or (content or "").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _tokenize_for_similarity(text: str) -> list:
    norm = normalize_text_for_comparison(text)
    words = [w for w in norm.split() if len(w) >= 2]
    # 한글/영문 문자 2-gram도 함께 넣어 띄어쓰기 차이에 강건하게 만듦
    compact = re.sub(r"\s+", "", norm)
    bigrams = [compact[i:i + 2] for i in range(len(compact) - 1)] if len(compact) >= 2 else []
    return words + bigrams


def compute_text_similarity(text_a: str, text_b: str) -> float:
    """두 기사 내용의 코사인 유사도(0.0 ~ 1.0)를 계산합니다."""
    tokens_a = _tokenize_for_similarity(text_a)
    tokens_b = _tokenize_for_similarity(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    vec_a = Counter(tokens_a)
    vec_b = Counter(tokens_b)

    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_vector_similarity(vec_a: list, vec_b: list) -> float:
    """선택적 임베딩 벡터 간 코사인 유사도를 계산합니다."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    nb = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def generate_issue_hash(keywords: list, title: str) -> str:
    """핵심 키워드 또는 제목 기준으로 이슈 해시를 생성합니다."""
    if keywords and len(keywords) >= 2:
        sorted_kw = "_".join(sorted([str(k).strip() for k in keywords[:3] if str(k).strip()]))
        if sorted_kw:
            return hashlib.sha256(sorted_kw.encode("utf-8")).hexdigest()
    clean_title = "".join(sorted((title or "").split()[:4]))
    return hashlib.sha256(clean_title.encode("utf-8")).hexdigest()


def get_or_create_master_id(cur, table_name: str, name: str) -> int:
    """categories 또는 sources 테이블에서 ID를 조회하고 없으면 생성 후 반환합니다."""
    clean_name = (name or "").strip() or "미분류"
    cur.execute(f"""
        INSERT INTO {table_name} (name) VALUES (%s)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id;
    """, (clean_name,))
    return cur.fetchone()[0]


def _find_matching_content_group(
    cur,
    c_hash: str,
    candidate_text: str,
    candidate_embedding: list = None,
    threshold: float = CONTENT_SIMILARITY_THRESHOLD,
):
    """
    기존 DB 기사들 중 내용이 같거나 매우 유사한 레코드를 찾습니다.
    반환: (matched_issue_id, matched_content_hash, similarity_score, match_reason) 또는 None
    """
    # 1. 정확 해시 일치
    if c_hash:
        cur.execute("""
            SELECT issue_id, content_hash
            FROM news_perspectives
            WHERE content_hash = %s
            ORDER BY id ASC
            LIMIT 1;
        """, (c_hash,))
        row = cur.fetchone()
        if row:
            return row[0], row[1], 1.0, "exact_content_hash"

    # 2. 최근 기사 본문/요약 비교기를 통한 유사도 매칭
    if not candidate_text or len(candidate_text.strip()) < 20:
        return None

    cur.execute("""
        SELECT
            p.issue_id,
            p.content_hash,
            COALESCE(NULLIF(p.raw_content, ''), NULLIF(p.summary_facts, ''), i.representative_summary, p.title) AS ref_text,
            p.metadata
        FROM news_perspectives p
        LEFT JOIN news_issues i ON i.id = p.issue_id
        ORDER BY p.id DESC
        LIMIT 200;
    """)
    candidates = cur.fetchall()

    best_match = None
    best_score = 0.0

    for issue_id, ref_hash, ref_text, meta in candidates:
        if not ref_text:
            continue
        score = compute_text_similarity(candidate_text, ref_text)

        # 메타데이터에 저장된 임베딩이 둘 다 있으면 임베딩 유사도도 반영
        if candidate_embedding and isinstance(meta, dict):
            ref_emb = meta.get("embedding")
            if isinstance(ref_emb, list) and ref_emb:
                emb_score = compute_vector_similarity(candidate_embedding, ref_emb)
                score = max(score, emb_score)

        if score >= threshold and score > best_score:
            best_score = score
            best_match = (issue_id, ref_hash, round(score, 4), f"comparator_sim>={threshold}")

    return best_match


def clean_old_news():
    """7일 경과 이슈 및 관점 데이터를 정리합니다."""
    conn = get_db_connection()
    cur = conn.cursor()
    week_ago = datetime.now() - timedelta(days=7)

    cur.execute(
        "DELETE FROM news_issues WHERE published_at < %s OR created_at < %s;",
        (week_ago, week_ago),
    )
    cur.execute(
        "DELETE FROM news_perspectives WHERE published_at < %s OR created_at < %s;",
        (week_ago, week_ago),
    )

    conn.commit()
    cur.close()
    conn.close()


def compact_old_raw_contents(days: int = 2):
    """완료된 지 일정 일수가 지난 기사의 원문(raw_content)을 비워 용량을 절약합니다."""
    conn = get_db_connection()
    cur = conn.cursor()
    cutoff = datetime.now() - timedelta(days=days)

    cur.execute("""
        UPDATE news_perspectives
        SET raw_content = NULL
        WHERE processing_status = 'COMPLETED'
          AND created_at < %s
          AND raw_content IS NOT NULL;
    """, (cutoff,))
    conn.commit()
    cur.close()
    conn.close()


def save_news_article(data: dict):
    """
    기사 내용이 중복이면(해시 일치 또는 비교기 유사도 초과):
    - 기사 내용(news_issues 신규 생성 및 raw_content 중복 저장)은 추가하지 않음
    - 기자 정보/의견(author_name, author_opinion, stance_type, url)만 news_perspectives에 추가/연계
    """
    conn = get_db_connection()
    cur = conn.cursor()

    raw = data.get("raw_content", "") or ""
    summary_facts = (data.get("summary_facts") or "").strip()
    comparison_text = raw.strip() if len(raw.strip()) >= 30 else summary_facts
    c_hash = generate_content_hash(comparison_text) if comparison_text else None
    candidate_embedding = data.get("embedding") or (data.get("metadata") or {}).get("embedding")

    keywords = data.get("keywords", []) or []
    title = (data.get("title") or "제목 없음").strip()
    published_at = data.get("published_at") or datetime.now()
    author_name = (data.get("author_name") or "기자 미상").strip()
    author_opinion = (data.get("author_opinion") or "").strip()
    stance_type = (data.get("stance_type") or "중립").strip()
    url = (data.get("url") or "").strip()

    try:
        category_id = get_or_create_master_id(cur, "categories", data.get("category", "종합"))
        source_id = get_or_create_master_id(cur, "sources", data.get("source", "기타언론"))

        matched = _find_matching_content_group(
            cur=cur,
            c_hash=c_hash,
            candidate_text=comparison_text,
            candidate_embedding=candidate_embedding,
        )

        # 1. 기사 내용이 중복인 경우 -> 기사 내용(이슈/본문) 추가 없이 기자 DB(news_perspectives)만 채우고 연계
        if matched:
            matched_issue_id, matched_hash, sim_score, match_reason = matched
            effective_hash = matched_hash or c_hash

            cur.execute("""
                UPDATE news_issues
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = %s;
            """, (matched_issue_id,))

            if url:
                cur.execute("""
                    SELECT id, author_opinion, author_name
                    FROM news_perspectives
                    WHERE url = %s
                       OR (issue_id = %s AND source_id = %s AND author_name = %s AND content_hash = %s)
                    LIMIT 1;
                """, (url, matched_issue_id, source_id, author_name, effective_hash))
                existing_persp = cur.fetchone()

                if existing_persp:
                    persp_id, existing_opinion, existing_author = existing_persp
                    new_author = author_name if (author_name and author_name != "기자 미상") else existing_author
                    new_opinion = author_opinion if author_opinion else (existing_opinion or "")
                    cur.execute("""
                        UPDATE news_perspectives
                        SET issue_id = %s,
                            source_id = COALESCE(%s, source_id),
                            author_name = %s,
                            author_opinion = %s,
                            stance_type = %s
                        WHERE id = %s;
                    """, (matched_issue_id, source_id, new_author, new_opinion, stance_type, persp_id))
                else:
                    merged_meta = dict(data.get("metadata") or {})
                    merged_meta.update({
                        "dedup_action": "linked_journalist_only_duplicate_content",
                        "match_reason": match_reason,
                        "similarity_score": sim_score,
                    })
                    # 본문 중복 방지를 위해 raw_content는 NULL로 두고 기존 이슈/해시와 연계
                    cur.execute("""
                        INSERT INTO news_perspectives (
                            issue_id, source_id, title, author_name, url,
                            summary_facts, author_opinion, stance_type,
                            raw_content, content_hash, thumbnail_url,
                            processing_status, model_version, metadata, published_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO UPDATE SET
                            issue_id = EXCLUDED.issue_id,
                            author_name = CASE
                                WHEN EXCLUDED.author_name <> '기자 미상' THEN EXCLUDED.author_name
                                ELSE news_perspectives.author_name
                            END,
                            author_opinion = CASE
                                WHEN EXCLUDED.author_opinion <> '' THEN EXCLUDED.author_opinion
                                ELSE news_perspectives.author_opinion
                            END,
                            stance_type = EXCLUDED.stance_type,
                            metadata = news_perspectives.metadata || EXCLUDED.metadata;
                    """, (
                        matched_issue_id,
                        source_id,
                        title,
                        author_name,
                        url,
                        summary_facts or None,
                        author_opinion,
                        stance_type,
                        effective_hash,
                        data.get("thumbnail_url", ""),
                        "COMPLETED",
                        data.get("model_version", "groq:openai/gpt-oss-120b"),
                        Json(merged_meta),
                        published_at,
                    ))

            conn.commit()
            return

        # 2. 새로운 기사 내용인 경우 -> 이슈 마스터 및 기자 관점 최초 적재
        i_hash = generate_issue_hash(keywords, title)
        cur.execute("""
            INSERT INTO news_issues (
                issue_title, representative_summary, category_id, keywords, issue_hash, published_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (issue_hash) DO UPDATE
                SET updated_at = CURRENT_TIMESTAMP,
                    representative_summary = COALESCE(news_issues.representative_summary, EXCLUDED.representative_summary)
            RETURNING id;
        """, (
            title,
            summary_facts or None,
            category_id,
            keywords,
            i_hash,
            published_at,
        ))
        issue_id = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO news_perspectives (
                issue_id, source_id, title, author_name, url,
                summary_facts, author_opinion, stance_type,
                raw_content, content_hash, thumbnail_url,
                processing_status, model_version, metadata, published_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                issue_id = EXCLUDED.issue_id,
                author_name = CASE
                    WHEN EXCLUDED.author_name <> '기자 미상' THEN EXCLUDED.author_name
                    ELSE news_perspectives.author_name
                END,
                author_opinion = CASE
                    WHEN EXCLUDED.author_opinion <> '' THEN EXCLUDED.author_opinion
                    ELSE news_perspectives.author_opinion
                END,
                stance_type = EXCLUDED.stance_type;
        """, (
            issue_id,
            source_id,
            title,
            author_name,
            url,
            summary_facts or None,
            author_opinion,
            stance_type,
            raw.strip() or None,
            c_hash,
            data.get("thumbnail_url", ""),
            "COMPLETED",
            data.get("model_version", "groq:openai/gpt-oss-120b"),
            Json(data.get("metadata", {})),
            published_at,
        ))

        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"이슈 및 기자 관점 저장 실패: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def url_exists(url: str) -> bool:
    """이미 news_perspectives에 적재된 기사 URL인지 확인합니다."""
    clean_url = (url or "").strip()
    if not clean_url:
        return False

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM news_perspectives WHERE url = %s LIMIT 1;",
            (clean_url,),
        )
        return cur.fetchone() is not None
    finally:
        cur.close()
        conn.close()



if __name__ == "__main__":
    print("=== 🚀 Neon PostgreSQL 데이터베이스 초기화 시작 ===")
    try:
        init_db()
        print("=== ✨ 데이터베이스 스키마(테이블 및 인덱스) 생성 완료! ===")
    except Exception as e:
        print(f"=== ❌ 데이터베이스 초기화 실패: {e} ===")