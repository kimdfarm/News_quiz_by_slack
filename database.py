import os
from datetime import datetime, timedelta
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    # 뉴스 저장 테이블 생성 (팩트와 저자 의견 분리 구조)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            summary_facts TEXT,
            author_opinion TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def clean_old_news():
    """1주일(7일)이 지난 뉴스 자동 삭제 (TTL 정책)"""
    conn = get_db_connection()
    cur = conn.cursor()
    week_ago = datetime.now() - timedelta(days=7)
    cur.execute("DELETE FROM news_articles WHERE created_at < %s;", (week_ago,))
    conn.commit()
    cur.close()
    conn.close()

def save_news(title, summary_facts, author_opinion, category):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO news_articles (title, summary_facts, author_opinion, category)
        VALUES (%s, %s, %s, %s);
    """, (title, summary_facts, author_opinion, category))
    conn.commit()
    cur.close()
    conn.close()