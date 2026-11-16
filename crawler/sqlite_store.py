import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

from database import (
    generate_content_hash,
    generate_issue_hash,
    compute_text_similarity,
    compute_vector_similarity,
    CONTENT_SIMILARITY_THRESHOLD,
)

DEFAULT_SQLITE_PATH = os.getenv("NEWS_SQLITE_PATH", "news_store.sqlite")


def get_sqlite_conn(db_path: str = DEFAULT_SQLITE_PATH, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True, timeout=15.0)
    else:
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_sqlite_db(db_path: str = DEFAULT_SQLITE_PATH):
    conn = get_sqlite_conn(db_path=db_path, read_only=False)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS news_issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_title TEXT NOT NULL,
            representative_summary TEXT,
            category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,
            keywords_json TEXT,
            issue_hash TEXT UNIQUE,
            published_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS news_perspectives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_id INTEGER REFERENCES news_issues(id) ON DELETE CASCADE,
            source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
            title TEXT NOT NULL,
            author_name TEXT,
            url TEXT UNIQUE NOT NULL,
            summary_facts TEXT,
            author_opinion TEXT,
            stance_type TEXT DEFAULT '중립',
            raw_content TEXT,
            content_hash TEXT,
            thumbnail_url TEXT,
            processing_status TEXT DEFAULT 'COMPLETED',
            model_version TEXT,
            metadata_json TEXT,
            published_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_sq_persp_issue ON news_perspectives(issue_id);
        CREATE INDEX IF NOT EXISTS idx_sq_persp_hash ON news_perspectives(content_hash);
        CREATE INDEX IF NOT EXISTS idx_sq_issues_pub ON news_issues(published_at DESC);
    """)
    conn.commit()
    cur.close()
    conn.close()


def get_existing_urls_sqlite(urls: List[str], db_path: str = DEFAULT_SQLITE_PATH) -> set:
    if not db_path:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SQLITE", "news_store.sqlite")
    if not os.path.exists(db_path):
        return set()
    clean_urls = list({(u or "").strip() for u in (urls or []) if (u or "").strip()})
    if not clean_urls:
        return set()

    conn = get_sqlite_conn(db_path=db_path, read_only=True)
    cur = conn.cursor()
    try:
        placeholders = ",".join(["?"] * len(clean_urls))
        cur.execute(f"SELECT url FROM news_perspectives WHERE url IN ({placeholders});", clean_urls)
        return {row["url"] for row in cur.fetchall() if row["url"]}
    finally:
        cur.close()
        conn.close()


def _get_or_create_master_id_sq(cur, table: str, name: str) -> int:
    clean = (name or "").strip() or ("미분류" if table == "categories" else "기타언론")
    cur.execute(f"INSERT OR IGNORE INTO {table} (name) VALUES (?);", (clean,))
    cur.execute(f"SELECT id FROM {table} WHERE name = ?;", (clean,))
    return cur.fetchone()["id"]


def _find_similar_content_sq(cur, c_hash: str, candidate_text: str, candidate_emb=None):
    if c_hash:
        cur.execute(
            "SELECT issue_id, content_hash FROM news_perspectives WHERE content_hash = ? ORDER BY id ASC LIMIT 1;",
            (c_hash,),
        )
        row = cur.fetchone()
        if row:
            return row["issue_id"], row["content_hash"], 1.0, "exact_content_hash"

    if not candidate_text or len(candidate_text.strip()) < 20:
        return None

    cur.execute("""
        SELECT p.issue_id, p.content_hash,
               COALESCE(NULLIF(p.raw_content, ''), NULLIF(p.summary_facts, ''), i.representative_summary, p.title) AS ref_text,
               p.metadata_json
        FROM news_perspectives p
        LEFT JOIN news_issues i ON i.id = p.issue_id
        ORDER BY p.id DESC LIMIT 200;
    """)
    best_match = None
    best_score = 0.0
    for r in cur.fetchall():
        ref_text = r["ref_text"] or ""
        if not ref_text:
            continue
        score = compute_text_similarity(candidate_text, ref_text)
        if candidate_emb and r["metadata_json"]:
            try:
                meta = json.loads(r["metadata_json"])
                if isinstance(meta.get("embedding"), list):
                    score = max(score, compute_vector_similarity(candidate_emb, meta["embedding"]))
            except Exception:
                pass
        if score >= CONTENT_SIMILARITY_THRESHOLD and score > best_score:
            best_score = score
            best_match = (r["issue_id"], r["content_hash"], round(score, 4), f"comparator>={CONTENT_SIMILARITY_THRESHOLD}")
    return best_match


def save_news_article_sqlite(data: Dict[str, Any], db_path: str = DEFAULT_SQLITE_PATH):
    if not db_path:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SQLITE", "news_store.sqlite")
    init_sqlite_db(db_path)
    conn = get_sqlite_conn(db_path=db_path, read_only=False)
    cur = conn.cursor()

    raw = data.get("raw_content", "") or ""
    summary_facts = (data.get("summary_facts") or "").strip()
    comparison_text = raw.strip() if len(raw.strip()) >= 30 else summary_facts
    c_hash = generate_content_hash(comparison_text) if comparison_text else None
    candidate_emb = data.get("embedding") or (data.get("metadata") or {}).get("embedding")

    keywords = data.get("keywords", []) or []
    title = (data.get("title") or "제목 없음").strip()
    pub_at = data.get("published_at")
    pub_str = pub_at.isoformat() if isinstance(pub_at, datetime) else str(pub_at or datetime.now().isoformat())
    author_name = (data.get("author_name") or "기자 미상").strip()
    author_opinion = (data.get("author_opinion") or "").strip()
    stance_type = (data.get("stance_type") or "중립").strip()
    url = (data.get("url") or "").strip()

    try:
        cat_id = _get_or_create_master_id_sq(cur, "categories", data.get("category", "종합"))
        src_id = _get_or_create_master_id_sq(cur, "sources", data.get("source", "기타언론"))

        matched = _find_similar_content_sq(cur, c_hash, comparison_text, candidate_emb)
        if matched:
            # 기사 내용은 추가하지 않고 기자 의견/관점만 news_perspectives에 연계
            matched_issue_id, matched_hash, sim_score, reason = matched
            eff_hash = matched_hash or c_hash
            cur.execute("UPDATE news_issues SET updated_at = CURRENT_TIMESTAMP WHERE id = ?;", (matched_issue_id,))
            if url:
                meta = dict(data.get("metadata") or {})
                meta.update({"linked_only": True, "match_reason": reason, "similarity": sim_score})
                cur.execute("""
                    INSERT INTO news_perspectives (
                        issue_id, source_id, title, author_name, url,
                        summary_facts, author_opinion, stance_type,
                        raw_content, content_hash, thumbnail_url,
                        processing_status, model_version, metadata_json, published_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 'COMPLETED', ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        issue_id = excluded.issue_id,
                        author_name = CASE WHEN excluded.author_name <> '기자 미상' THEN excluded.author_name ELSE news_perspectives.author_name END,
                        author_opinion = CASE WHEN excluded.author_opinion <> '' THEN excluded.author_opinion ELSE news_perspectives.author_opinion END,
                        stance_type = excluded.stance_type;
                """, (
                    matched_issue_id, src_id, title, author_name, url,
                    summary_facts or None, author_opinion, stance_type,
                    eff_hash, data.get("thumbnail_url", ""),
                    data.get("model_version", ""), json.dumps(meta, ensure_ascii=False), pub_str,
                ))
            conn.commit()
            return

        i_hash = generate_issue_hash(keywords, title)
        cur.execute("""
            INSERT INTO news_issues (
                issue_title, representative_summary, category_id, keywords_json, issue_hash, published_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(issue_hash) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP,
                representative_summary = COALESCE(news_issues.representative_summary, excluded.representative_summary);
        """, (title, summary_facts or None, cat_id, json.dumps(keywords, ensure_ascii=False), i_hash, pub_str))

        cur.execute("SELECT id FROM news_issues WHERE issue_hash = ?;", (i_hash,))
        issue_id = cur.fetchone()["id"]

        cur.execute("""
            INSERT INTO news_perspectives (
                issue_id, source_id, title, author_name, url,
                summary_facts, author_opinion, stance_type,
                raw_content, content_hash, thumbnail_url,
                processing_status, model_version, metadata_json, published_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED', ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                issue_id = excluded.issue_id,
                author_name = CASE WHEN excluded.author_name <> '기자 미상' THEN excluded.author_name ELSE news_perspectives.author_name END,
                author_opinion = CASE WHEN excluded.author_opinion <> '' THEN excluded.author_opinion ELSE news_perspectives.author_opinion END,
                stance_type = excluded.stance_type;
        """, (
            issue_id, src_id, title, author_name, url,
            summary_facts or None, author_opinion, stance_type,
            raw.strip() or None, c_hash, data.get("thumbnail_url", ""),
            data.get("model_version", ""), json.dumps(data.get("metadata") or {}, ensure_ascii=False), pub_str,
        ))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

def update_perspective_sqlite(
    perspective_id: int,
    updates: Dict[str, Any],
    db_path: str = DEFAULT_SQLITE_PATH,
) -> int:
    """특정 기자 관점(news_perspectives) 레코드의 일부 필드를 수정하고 파일에 저장합니다."""
    allowed_fields = {
        "title",
        "author_name",
        "summary_facts",
        "author_opinion",
        "stance_type",
        "issue_id",
        "processing_status",
    }
    clean_updates = {k: v for k, v in (updates or {}).items() if k in allowed_fields}
    if not clean_updates:
        return 0

    set_clause = ", ".join([f"{col} = ?" for col in clean_updates.keys()])
    params = list(clean_updates.values()) + [perspective_id]

    conn = get_sqlite_conn(db_path=db_path, read_only=False)
    cur = conn.cursor()
    try:
        cur.execute(f"UPDATE news_perspectives SET {set_clause} WHERE id = ?;", params)
        affected = cur.rowcount
        conn.commit()
        return affected
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def delete_perspectives_by_ids_sqlite(
    perspective_ids: List[int],
    db_path: str = DEFAULT_SQLITE_PATH,
) -> int:
    """선택한 ID 목록의 기사/기자 관점만 일부 삭제합니다."""
    clean_ids = [int(i) for i in (perspective_ids or []) if str(i).isdigit()]
    if not clean_ids:
        return 0

    conn = get_sqlite_conn(db_path=db_path, read_only=False)
    cur = conn.cursor()
    try:
        placeholders = ",".join(["?"] * len(clean_ids))
        cur.execute(f"DELETE FROM news_perspectives WHERE id IN ({placeholders});", clean_ids)
        deleted = cur.rowcount
        # 연결된 관점이 하나도 남은 이슈는 같이 정리하고 싶을 때
        cur.execute("""
            DELETE FROM news_issues
            WHERE id NOT IN (SELECT DISTINCT issue_id FROM news_perspectives WHERE issue_id IS NOT NULL);
        """)
        conn.commit()
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def delete_old_records_sqlite(
    older_than_iso: str,
    db_path: str = DEFAULT_SQLITE_PATH,
) -> int:
    """특정 기준 시각(ISO 문자열) 이전의 오래된 데이터를 일부 삭제합니다."""
    conn = get_sqlite_conn(db_path=db_path, read_only=False)
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM news_perspectives WHERE published_at < ?;", (older_than_iso,))
        deleted = cur.rowcount
        cur.execute("""
            DELETE FROM news_issues
            WHERE id NOT IN (SELECT DISTINCT issue_id FROM news_perspectives WHERE issue_id IS NOT NULL);
        """)
        conn.commit()
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def clear_all_sqlite_data(
    db_path: str = DEFAULT_SQLITE_PATH,
    vacuum: bool = True,
    recreate_file: bool = False,
):
    """전체 데이터를 삭제합니다. recreate_file=True면 파일 자체를 지우고 스키마만 새로 만듭니다."""
    if recreate_file:
        for suffix in ("", "-wal", "-shm"):
            target = f"{db_path}{suffix}"
            if os.path.exists(target):
                os.remove(target)
        init_sqlite_db(db_path)
        return

    if not os.path.exists(db_path):
        init_sqlite_db(db_path)
        return

    conn = get_sqlite_conn(db_path=db_path, read_only=False)
    cur = conn.cursor()
    try:
        cur.executescript("""
            DELETE FROM news_perspectives;
            DELETE FROM news_issues;
            DELETE FROM categories;
            DELETE FROM sources;
            DELETE FROM sqlite_sequence WHERE name IN ('news_perspectives', 'news_issues', 'categories', 'sources');
        """)
        conn.commit()
        if vacuum:
            # 삭제 후 줄어든 용량을 실제 파일 크기에도 반영
            conn.execute("VACUUM;")
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()