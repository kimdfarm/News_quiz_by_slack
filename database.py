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
    2. news_issues (이슈 마스터 - 사건 단위 클러스터링 및 벡터 검색)
    3. news_perspectives (언론사/기자별 개별 논조 및 상세 데이터)
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    # pgvector 확장 활성화 (DB 권한에 따라 실패할 수 있으므로 예외 처리)
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"pgvector 확장 로드 스킵 (기본 해시 매칭으로 동작): {e}")

    cur.execute("""
        -- 1. 카테고리 마스터 (문자열 반복 제거로 공간 절약)
        CREATE TABLE IF NOT EXISTS categories (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        );

        -- 2. 언론사 마스터 (문자열 반복 제거로 공간 절약)
        CREATE TABLE IF NOT EXISTS sources (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        );

        -- 3. 이슈 마스터 테이블 (동일 사건/주제당 1개 레코드만 유지)
        CREATE TABLE IF NOT EXISTS news_issues (
            id SERIAL PRIMARY KEY,
            issue_title TEXT NOT NULL,              -- 대표 이슈 제목
            representative_summary TEXT,            -- 사건의 객관적 팩트 종합 요약
            category_id INT REFERENCES categories(id) ON DELETE SET NULL,
            keywords TEXT[],                        -- 핵심 키워드 배열
            issue_hash VARCHAR(64) UNIQUE,          -- 이슈 중복 방지용 해시 (키워드/제목 정규화 기반)
            published_at TIMESTAMP,                 -- 최초 보도 일시
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 4. 언론사별/기자별 관점 상세 테이블 (하나의 이슈에 여러 언론사 시각을 N개로 묶음)
        CREATE TABLE IF NOT EXISTS news_perspectives (
            id SERIAL PRIMARY KEY,
            issue_id INT REFERENCES news_issues(id) ON DELETE CASCADE,
            source_id INT REFERENCES sources(id) ON DELETE SET NULL,
            
            title TEXT NOT NULL,                    -- 해당 언론사의 기사 제목
            author_name TEXT,                       -- 기자 이름
            url TEXT UNIQUE NOT NULL,               -- 원본 기사 URL (1차 중복 방지)
            
            -- 핵심: 언론사/기자마다 다른 생각과 논조 분리
            summary_facts TEXT,                     -- 해당 기사에서 추출한 팩트
            author_opinion TEXT,                    -- 기자/언론사만의 주관적 논평 및 해석
            stance_type VARCHAR(20) DEFAULT '중립', -- 논조 (긍정, 부정, 비판, 중립 등)
            
            -- 원문 및 중복 제어
            raw_content TEXT,                       -- 기사 원문 (추후 용량 다이어트 시 NULL 처리 가능)
            content_hash VARCHAR(64),               -- 본문 SHA-256 해시 (복붙 기사 2차 필터링)
            thumbnail_url TEXT,
            
            processing_status VARCHAR(20) DEFAULT 'COMPLETED',
            model_version VARCHAR(50),
            metadata JSONB,
            
            published_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 검색, 조인 및 중복 방지 인덱스
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
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def generate_issue_hash(keywords: list, title: str) -> str:
    """
    동일한 사건(이슈)인지 판별하기 위한 정규화 해시 생성.
    핵심 키워드(상위 3개)를 정렬하여 조합함으로써 언론사마다 제목이 조금 달라도 같은 이슈로 묶어줍니다.
    """
    if keywords and len(keywords) >= 2:
        sorted_kw = "_".join(sorted([k.strip() for k in keywords[:3]]))
        return hashlib.sha256(sorted_kw.encode('utf-8')).hexdigest()
    # 키워드가 부족할 경우 제목의 핵심 어절 기준 해시 생성
    clean_title = "".join(sorted(title.split()[:4]))
    return hashlib.sha256(clean_title.encode('utf-8')).hexdigest()

def get_or_create_master_id(cur, table_name: str, name: str) -> int:
    """categories 또는 sources 테이블에서 ID를 조회하고 없으면 생성 후 반환합니다."""
    if not name:
        name = "미분류"
    cur.execute(f"""
        INSERT INTO {table_name} (name) VALUES (%s)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id;
    """, (name.strip(),))
    return cur.fetchone()[0]

def clean_old_news():
    """
    슬라이딩 윈도우 TTL 정책 (1주일):
    1주일이 경과한 이슈(news_issues)를 삭제하면 ON DELETE CASCADE에 의해
    하위 언론사별 관점(news_perspectives)도 한 번에 깔끔하게 삭제됩니다.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    week_ago = datetime.now() - timedelta(days=7)
    
    cur.execute("DELETE FROM news_issues WHERE published_at < %s OR created_at < %s;", (week_ago, week_ago))
    # 혹시 고아 레코드가 남았을 경우 대비
    cur.execute("DELETE FROM news_perspectives WHERE published_at < %s OR created_at < %s;", (week_ago, week_ago))
    
    conn.commit()
    cur.close()
    conn.close()

def compact_old_raw_contents(days: int = 2):
    """
    [공간 절약용 유지보수 함수]
    이미 LLM 요약/분석(COMPLETED)이 끝난 지 2일이 지난 기사는 
    무거운 원문(raw_content)만 비워서 DB 용량을 획기적으로 줄입니다.
    """
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
    크롤링 및 LLM 분석이 끝난 데이터를 이슈별로 클러스터링하여 적재합니다.
    - 같은 이슈(issue_hash)가 이미 존재하면 기존 issue_id에 언론사/기자 관점만 추가합니다.
    - 완전 복붙 기사(동일 이슈 내 동일 content_hash)는 중복 적재를 차단합니다.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    raw = data.get("raw_content", "")
    c_hash = generate_content_hash(raw) if raw else None
    keywords = data.get("keywords", [])
    title = data.get("title", "제목 없음")
    published_at = data.get("published_at") or datetime.now()
    
    try:
        # 1. 마스터 테이블(카테고리, 언론사) 정규화 ID 확보
        category_id = get_or_create_master_id(cur, "categories", data.get("category", "종합"))
        source_id = get_or_create_master_id(cur, "sources", data.get("source", "기타언론"))
        
        # 2. 이슈 해시 생성 및 이슈 마스터(news_issues) 조회/생성
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
            published_at
        ))
        issue_id = cur.fetchone()[0]
        
        # 3. 동일 이슈 내에 본문 해시(content_hash)가 완전히 똑같은 복붙 기사가 이미 있는지 체크
        if c_hash:
            cur.execute("""
                SELECT id FROM news_perspectives 
                WHERE issue_id = %s AND content_hash = %s LIMIT 1;
            """, (issue_id, c_hash))
            if cur.fetchone():
                # 내용이 100% 동일한 복붙 기사는 저장하지 않고 스킵
                conn.commit()
                return

        # 4. 언론사/기자별 개별 관점(news_perspectives) 적재
        cur.execute("""
            INSERT INTO news_perspectives (
                issue_id, source_id, title, author_name, url,
                summary_facts, author_opinion, stance_type,
                raw_content, content_hash, thumbnail_url,
                processing_status, model_version, metadata, published_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING;
        """, (
            issue_id,
            source_id,
            title,
            data.get("author_name", "기자 미상"),
            data.get("url"),
            data.get("summary_facts"),
            data.get("author_opinion", ""),
            data.get("stance_type", "중립"),
            raw,
            c_hash,
            data.get("thumbnail_url", ""),
            "COMPLETED",
            data.get("model_version", "gemma:3.4b"),
            Json(data.get("metadata", {})),
            published_at
        ))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"이슈 및 관점 저장 실패: {e}")
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