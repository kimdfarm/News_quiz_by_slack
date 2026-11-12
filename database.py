import os
import hashlib
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

# .env 파일에서 환경 변수(DATABASE_URL 등) 로드
load_dotenv()

def get_db_connection():
    """PostgreSQL 데이터베이스 연결을 생성하고 반환합니다."""
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def init_db():
    """
    고도화된 스키마(news_articles)를 바탕으로 
    테이블 및 성능 최적화를 위한 인덱스들을 초기화(생성)합니다.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        -- 뉴스 기사 저장 메인 테이블
        CREATE TABLE IF NOT EXISTS news_articles (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            
            -- 1. 팩트와 의견 분리 및 핵심 데이터
            summary_facts TEXT,                -- 객관적 팩트 요약
            author_opinion TEXT,               -- 저자/기자 의견 및 주관
            category TEXT,                     -- 카테고리 (정치, 경제, IT 등)
            keywords TEXT[],                   -- 주요 키워드 배열 (PostgreSQL 배열 타입)
            
            -- 2. 메타데이터 및 출처 관리
            source TEXT,                       -- 언론사 / 출처 (예: 연합뉴스)
            url TEXT UNIQUE,                   -- 원본 기사 URL (UNIQUE 제약으로 중복 수집 1차 방지)
            author_name TEXT,                  -- 기자 이름
            published_at TIMESTAMP,            -- 실제 기사 보도 일시
            thumbnail_url TEXT,                -- 기사 대표 이미지 (UI용)
            
            -- 3. 원문 보존 및 파이프라인 최적화
            raw_content TEXT,                  -- 기사 원문 전체 (재파싱/재요약 대비)
            content_hash VARCHAR(64),          -- 원문 SHA-256 해시값 (동일 기사 중복 수집 2차 방지)
            processing_status VARCHAR(20) DEFAULT 'PENDING', -- 파이프라인 상태 (PENDING, COMPLETED, FAILED)
            model_version VARCHAR(50),         -- 요약/추출에 사용된 LLM 모델 또는 프롬프트 버전
            
            -- 4. 유연한 확장을 위한 메타데이터 저장소 (논조 점수, 감성 분석 결과 등 자유롭게 보관)
            metadata JSONB,                    
            
            -- 5. 시스템 관리용
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 검색, 정렬 및 중복 방지 성능을 위한 인덱스 생성
        CREATE INDEX IF NOT EXISTS idx_news_published_at ON news_articles(published_at DESC);
        CREATE INDEX IF NOT EXISTS idx_news_category ON news_articles(category);
        CREATE INDEX IF NOT EXISTS idx_news_source ON news_articles(source);
        CREATE INDEX IF NOT EXISTS idx_news_content_hash ON news_articles(content_hash); -- 해시 기반 중복 체크 고속화
        CREATE INDEX IF NOT EXISTS idx_news_status ON news_articles(processing_status);   -- 미처리 데이터 배치 작업용
    """)
    conn.commit()
    cur.close()
    conn.close()

def generate_content_hash(content: str) -> str:
    """기사 원문 텍스트를 기반으로 SHA-256 해시값을 생성합니다. (중복 수집 2차 방지용)"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def clean_old_news():
    """
    슬라이딩 윈도우 TTL 정책 적용:
    보도 일시(published_at)나 생성 일시(created_at) 기준으로 
    1주일(7일)이 경과한 뉴스 데이터를 자동으로 삭제하여 DB 용량을 관리합니다.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    week_ago = datetime.now() - timedelta(days=7)
    
    # 7일 이전 데이터 삭제 쿼리 실행
    cur.execute("DELETE FROM news_articles WHERE published_at < %s OR created_at < %s;", (week_ago, week_ago))
    conn.commit()
    cur.close()
    conn.close()

def save_news_article(data: dict):
    """
    파싱된 뉴스 기사 데이터를 데이터베이스에 안전하게 적재합니다.
    URL이 중복될 경우(ON CONFLICT) 저장을 무시하여 1차 중복을 방지합니다.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    raw = data.get("raw_content", "")
    c_hash = generate_content_hash(raw) if raw else None
    
    try:
        cur.execute("""
            INSERT INTO news_articles (
                title, summary_facts, author_opinion, category, keywords,
                source, url, author_name, published_at, thumbnail_url,
                raw_content, content_hash, processing_status, model_version, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING; -- URL이 이미 존재하면 INSERT 무시
        """, (
            data.get("title"),
            data.get("summary_facts"),
            data.get("author_opinion"),
            data.get("category"),
            data.get("keywords", []),
            data.get("source"),
            data.get("url"),
            data.get("author_name"),
            data.get("published_at"),
            data.get("thumbnail_url"),
            raw,
            c_hash,
            "COMPLETED", # 정상 처리 완료 상태로 설정
            data.get("model_version", "gemma:3.4b"),
            Json(data.get("metadata", {}))
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"기사 저장 실패: {e}")
    finally:
        cur.close()
        conn.close()