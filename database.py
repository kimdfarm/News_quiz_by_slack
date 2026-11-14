import os
import hashlib
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """PostgreSQL 데이터베이스 연결을 생성하고 반환합니다."""
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def init_db():
    """
    이슈 중심 정규화 스키마 초기화:
    1. categories, sources (마스터 테이블 - 공간 절약)
    2. news_issues (이슈/기사 내용 마스터)
    3. news_perspectives (언론사/기자별 의견 및 관점 상세)
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"pgvector 확장 로드 스킵 (기본 해시 매칭으로 동작): {e}")

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


def generate_content_hash(content: str) -> str:
    """기사 원문 텍스트를 기반으로 SHA-256 해시값을 생성합니다."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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
    기사 내용(content_hash)이 이미 있으면 기사 원문/내용을 중복 저장하지 않되,
    기자 의견(author_opinion)이나 기자 관점 정보가 있으면 news_perspectives에 추가합니다.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    raw = data.get("raw_content", "") or ""
    c_hash = generate_content_hash(raw) if raw.strip() else None
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
        i_hash = generate_issue_hash(keywords, title)

        cur.execute("""
            INSERT INTO news_issues (
                issue_title, representative_summary, category_id, keywords, issue_hash, published_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (issue_hash) DO UPDATE
                SET updated_at = CURRENT_TIMESTAMP
            RETURNING id;
        """, (
            title,
            data.get("summary_facts"),
            category_id,
            keywords,
            i_hash,
            published_at,
        ))
        issue_id = cur.fetchone()[0]

        # 1. 동일한 기사 내용(content_hash)이 이미 존재하는지 확인
        existing_same_content = None
        if c_hash:
            cur.execute("""
                SELECT id, issue_id, author_opinion
                FROM news_perspectives
                WHERE content_hash = %s
                ORDER BY id ASC
                LIMIT 1;
            """, (c_hash,))
            existing_same_content = cur.fetchone()

        if existing_same_content:
            _, existing_issue_id, _ = existing_same_content
            target_issue_id = existing_issue_id or issue_id

            # 같은 기사 내용이더라도 새로운 기자 의견이나 다른 기자/언론사 관점이 있으면 추가
            has_journalist_info = bool(author_opinion) or (author_name not in ("", "기자 미상"))
            if has_journalist_info and url:
                cur.execute("""
                    SELECT id, author_opinion FROM news_perspectives
                    WHERE url = %s
                       OR (issue_id = %s AND source_id = %s AND author_name = %s AND content_hash = %s)
                    LIMIT 1;
                """, (url, target_issue_id, source_id, author_name, c_hash))
                dup_perspective = cur.fetchone()

                if dup_perspective:
                    dup_id, existing_opinion = dup_perspective
                    if author_opinion and not (existing_opinion or "").strip():
                        cur.execute("""
                            UPDATE news_perspectives
                            SET author_opinion = %s, stance_type = %s
                            WHERE id = %s;
                        """, (author_opinion, stance_type, dup_id))
                else:
                    # 기사 원문(raw_content)은 이미 존재하므로 NULL로 두어 본문 중복 저장 방지
                    cur.execute("""
                        INSERT INTO news_perspectives (
                            issue_id, source_id, title, author_name, url,
                            summary_facts, author_opinion, stance_type,
                            raw_content, content_hash, thumbnail_url,
                            processing_status, model_version, metadata, published_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO UPDATE SET
                            author_opinion = CASE
                                WHEN (news_perspectives.author_opinion IS NULL OR news_perspectives.author_opinion = '')
                                     AND EXCLUDED.author_opinion <> ''
                                THEN EXCLUDED.author_opinion
                                ELSE news_perspectives.author_opinion
                            END,
                            stance_type = COALESCE(EXCLUDED.stance_type, news_perspectives.stance_type);
                    """, (
                        target_issue_id,
                        source_id,
                        title,
                        author_name,
                        url,
                        data.get("summary_facts"),
                        author_opinion,
                        stance_type,
                        c_hash,
                        data.get("thumbnail_url", ""),
                        "COMPLETED",
                        data.get("model_version", "gemma:3.4b"),
                        Json(data.get("metadata", {})),
                        published_at,
                    ))

            conn.commit()
            return

        # 2. 새로운 기사 내용인 경우 news_perspectives에 본문 포함 정상 적재
        cur.execute("""
            INSERT INTO news_perspectives (
                issue_id, source_id, title, author_name, url,
                summary_facts, author_opinion, stance_type,
                raw_content, content_hash, thumbnail_url,
                processing_status, model_version, metadata, published_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                author_opinion = CASE
                    WHEN (news_perspectives.author_opinion IS NULL OR news_perspectives.author_opinion = '')
                         AND EXCLUDED.author_opinion <> ''
                    THEN EXCLUDED.author_opinion
                    ELSE news_perspectives.author_opinion
                END;
        """, (
            issue_id,
            source_id,
            title,
            author_name,
            url,
            data.get("summary_facts"),
            author_opinion,
            stance_type,
            raw if raw.strip() else None,
            c_hash,
            data.get("thumbnail_url", ""),
            "COMPLETED",
            data.get("model_version", "gemma:3.4b"),
            Json(data.get("metadata", {})),
            published_at,
        ))

        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"이슈 및 관점 저장 실패: {e}")
        raise
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