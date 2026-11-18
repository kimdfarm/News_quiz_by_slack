import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from database import (
    CONTENT_SIMILARITY_THRESHOLD,
    compute_text_similarity,
    compute_vector_similarity,
    generate_content_hash,
    generate_issue_hash,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_SQLITE_DIR = os.path.join(PROJECT_ROOT, "SQLITE")
DEFAULT_SQLITE_PATH = os.getenv(
    "NEWS_SQLITE_PATH",
    os.path.join(DEFAULT_SQLITE_DIR, "news_store.sqlite"),
)


def _resolve_db_path(db_path: Optional[str] = None) -> str:
    cleaned = (db_path or "").strip()
    return cleaned if cleaned else DEFAULT_SQLITE_PATH


def get_sqlite_conn(db_path: Optional[str] = None, read_only: bool = False) -> sqlite3.Connection:
    target_path = _resolve_db_path(db_path)
    if read_only:
        conn = sqlite3.connect(f"file:{os.path.abspath(target_path)}?mode=ro", uri=True, timeout=15.0)
    else:
        parent = os.path.dirname(os.path.abspath(target_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(target_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_sqlite_db(db_path: Optional[str] = None):
    """database.py와 동일한 테이블·컬럼·인덱스 구조로 SQLite DB를 초기화합니다."""
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
            keywords TEXT,
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
            metadata TEXT,

            published_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
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


def _normalize_dt_str(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if value:
        raw = str(value).strip()
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return raw
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_json_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def url_exists_sqlite(url: str, db_path: Optional[str] = None) -> bool:
    """database.py의 url_exists와 동일하게 URL 적재 여부를 확인합니다."""
    clean_url = (url or "").strip()
    target_path = _resolve_db_path(db_path)
    if not clean_url or not os.path.exists(target_path):
        return False

    conn = get_sqlite_conn(db_path=target_path, read_only=True)
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM news_perspectives WHERE url = ? LIMIT 1;", (clean_url,))
        return cur.fetchone() is not None
    finally:
        cur.close()
        conn.close()


def get_existing_urls_sqlite(urls: List[str], db_path: Optional[str] = None) -> set:
    target_path = _resolve_db_path(db_path)
    if not os.path.exists(target_path):
        return set()
    clean_urls = list({(u or "").strip() for u in (urls or []) if (u or "").strip()})
    if not clean_urls:
        return set()

    conn = get_sqlite_conn(db_path=target_path, read_only=True)
    cur = conn.cursor()
    try:
        placeholders = ",".join(["?"] * len(clean_urls))
        cur.execute(f"SELECT url FROM news_perspectives WHERE url IN ({placeholders});", clean_urls)
        return {row["url"] for row in cur.fetchall() if row["url"]}
    finally:
        cur.close()
        conn.close()


def _get_or_create_master_id_sq(cur, table: str, name: str) -> int:
    clean_name = (name or "").strip() or "미분류"
    cur.execute(
        f"""
        INSERT INTO {table} (name) VALUES (?)
        ON CONFLICT(name) DO UPDATE SET name = excluded.name;
        """,
        (clean_name,),
    )
    cur.execute(f"SELECT id FROM {table} WHERE name = ?;", (clean_name,))
    return cur.fetchone()["id"]


def _find_matching_content_group_sq(
    cur,
    c_hash: Optional[str],
    candidate_text: str,
    candidate_embedding: Optional[list] = None,
    threshold: float = CONTENT_SIMILARITY_THRESHOLD,
):
    """database.py의 _find_matching_content_group과 동일한 2단계(해시 -> 유사도) 매칭을 수행합니다."""
    if c_hash:
        cur.execute(
            """
            SELECT issue_id, content_hash
            FROM news_perspectives
            WHERE content_hash = ?
            ORDER BY id ASC
            LIMIT 1;
            """,
            (c_hash,),
        )
        row = cur.fetchone()
        if row:
            return row["issue_id"], row["content_hash"], 1.0, "exact_content_hash"

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

    for row in candidates:
        ref_text = row["ref_text"] or ""
        if not ref_text:
            continue
        score = compute_text_similarity(candidate_text, ref_text)

        if candidate_embedding and row["metadata"]:
            meta = _parse_json_dict(row["metadata"])
            ref_emb = meta.get("embedding")
            if isinstance(ref_emb, list) and ref_emb:
                emb_score = compute_vector_similarity(candidate_embedding, ref_emb)
                score = max(score, emb_score)

        if score >= threshold and score > best_score:
            best_score = score
            best_match = (
                row["issue_id"],
                row["content_hash"],
                round(score, 4),
                f"comparator_sim>={threshold}",
            )

    return best_match


def save_news_article_sqlite(data: Dict[str, Any], db_path: Optional[str] = None):
    """database.py의 save_news_article과 동일한 규칙으로 SQLite에 이슈 및 기자 관점을 적재합니다."""
    target_path = _resolve_db_path(db_path)
    init_sqlite_db(target_path)
    conn = get_sqlite_conn(db_path=target_path, read_only=False)
    cur = conn.cursor()

    raw = data.get("raw_content", "") or ""
    summary_facts = (data.get("summary_facts") or "").strip()
    comparison_text = raw.strip() if len(raw.strip()) >= 30 else summary_facts
    c_hash = generate_content_hash(comparison_text) if comparison_text else None
    candidate_embedding = data.get("embedding") or (data.get("metadata") or {}).get("embedding")

    keywords = data.get("keywords", []) or []
    title = (data.get("title") or "제목 없음").strip()
    published_at = _normalize_dt_str(data.get("published_at"))
    author_name = (data.get("author_name") or "기자 미상").strip()
    author_opinion = (data.get("author_opinion") or "").strip()
    stance_type = (data.get("stance_type") or "중립").strip()
    url = (data.get("url") or "").strip()

    try:
        category_id = _get_or_create_master_id_sq(cur, "categories", data.get("category", "종합"))
        source_id = _get_or_create_master_id_sq(cur, "sources", data.get("source", "기타언론"))

        matched = _find_matching_content_group_sq(
            cur=cur,
            c_hash=c_hash,
            candidate_text=comparison_text,
            candidate_embedding=candidate_embedding,
        )

        # 1. 기사 내용이 중복인 경우 -> 신규 이슈/본문 추가 없이 기존 이슈에 기자 관점만 연계
        if matched:
            matched_issue_id, matched_hash, sim_score, match_reason = matched
            effective_hash = matched_hash or c_hash

            cur.execute(
                """
                UPDATE news_issues
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?;
                """,
                (matched_issue_id,),
            )

            if url:
                cur.execute(
                    """
                    SELECT id, author_opinion, author_name, metadata
                    FROM news_perspectives
                    WHERE url = ?
                       OR (issue_id = ? AND source_id = ? AND author_name = ? AND content_hash = ?)
                    LIMIT 1;
                    """,
                    (url, matched_issue_id, source_id, author_name, effective_hash),
                )
                existing_persp = cur.fetchone()

                merged_meta = _parse_json_dict(data.get("metadata"))
                merged_meta.update({
                    "dedup_action": "linked_journalist_only_duplicate_content",
                    "match_reason": match_reason,
                    "similarity_score": sim_score,
                })

                if existing_persp:
                    persp_id = existing_persp["id"]
                    existing_opinion = existing_persp["author_opinion"]
                    existing_author = existing_persp["author_name"]
                    prev_meta = _parse_json_dict(existing_persp["metadata"])
                    prev_meta.update(merged_meta)

                    new_author = author_name if (author_name and author_name != "기자 미상") else existing_author
                    new_opinion = author_opinion if author_opinion else (existing_opinion or "")
                    cur.execute(
                        """
                        UPDATE news_perspectives
                        SET issue_id = ?,
                            source_id = COALESCE(?, source_id),
                            author_name = ?,
                            author_opinion = ?,
                            stance_type = ?,
                            metadata = ?
                        WHERE id = ?;
                        """,
                        (
                            matched_issue_id,
                            source_id,
                            new_author,
                            new_opinion,
                            stance_type,
                            json.dumps(prev_meta, ensure_ascii=False),
                            persp_id,
                        ),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO news_perspectives (
                            issue_id, source_id, title, author_name, url,
                            summary_facts, author_opinion, stance_type,
                            raw_content, content_hash, thumbnail_url,
                            processing_status, model_version, metadata, published_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(url) DO UPDATE SET
                            issue_id = excluded.issue_id,
                            author_name = CASE
                                WHEN excluded.author_name <> '기자 미상' THEN excluded.author_name
                                ELSE news_perspectives.author_name
                            END,
                            author_opinion = CASE
                                WHEN excluded.author_opinion <> '' THEN excluded.author_opinion
                                ELSE news_perspectives.author_opinion
                            END,
                            stance_type = excluded.stance_type,
                            metadata = excluded.metadata;
                        """,
                        (
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
                            json.dumps(merged_meta, ensure_ascii=False),
                            published_at,
                        ),
                    )

            conn.commit()
            return

        # 2. 새로운 기사 내용인 경우 -> 이슈 마스터 및 기자 관점 최초 적재
        i_hash = generate_issue_hash(keywords, title)
        cur.execute(
            """
            INSERT INTO news_issues (
                issue_title, representative_summary, category_id, keywords, issue_hash, published_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(issue_hash) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP,
                representative_summary = COALESCE(news_issues.representative_summary, excluded.representative_summary);
            """,
            (
                title,
                summary_facts or None,
                category_id,
                json.dumps(keywords, ensure_ascii=False),
                i_hash,
                published_at,
            ),
        )
        cur.execute("SELECT id FROM news_issues WHERE issue_hash = ?;", (i_hash,))
        issue_id = cur.fetchone()["id"]

        cur.execute(
            """
            INSERT INTO news_perspectives (
                issue_id, source_id, title, author_name, url,
                summary_facts, author_opinion, stance_type,
                raw_content, content_hash, thumbnail_url,
                processing_status, model_version, metadata, published_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                issue_id = excluded.issue_id,
                author_name = CASE
                    WHEN excluded.author_name <> '기자 미상' THEN excluded.author_name
                    ELSE news_perspectives.author_name
                END,
                author_opinion = CASE
                    WHEN excluded.author_opinion <> '' THEN excluded.author_opinion
                    ELSE news_perspectives.author_opinion
                END,
                stance_type = excluded.stance_type;
            """,
            (
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
                json.dumps(_parse_json_dict(data.get("metadata")), ensure_ascii=False),
                published_at,
            ),
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def clean_old_news_sqlite(days: int = 7, db_path: Optional[str] = None):
    """database.py의 clean_old_news와 동일하게 7일 경과 데이터를 정리합니다."""
    target_path = _resolve_db_path(db_path)
    if not os.path.exists(target_path):
        return

    conn = get_sqlite_conn(db_path=target_path, read_only=False)
    cur = conn.cursor()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        cur.execute(
            "DELETE FROM news_issues WHERE published_at < ? OR created_at < ?;",
            (cutoff, cutoff),
        )
        cur.execute(
            "DELETE FROM news_perspectives WHERE published_at < ? OR created_at < ?;",
            (cutoff, cutoff),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def compact_old_raw_contents_sqlite(days: int = 2, db_path: Optional[str] = None):
    """database.py의 compact_old_raw_contents와 동일하게 오래된 원문을 비워 용량을 절약합니다."""
    target_path = _resolve_db_path(db_path)
    if not os.path.exists(target_path):
        return

    conn = get_sqlite_conn(db_path=target_path, read_only=False)
    cur = conn.cursor()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        cur.execute(
            """
            UPDATE news_perspectives
            SET raw_content = NULL
            WHERE processing_status = 'COMPLETED'
              AND created_at < ?
              AND raw_content IS NOT NULL;
            """,
            (cutoff,),
        )
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
    db_path: Optional[str] = None,
) -> int:
    """특정 기자 관점(news_perspectives) 레코드의 일부 필드를 수정합니다."""
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

    conn = get_sqlite_conn(db_path=_resolve_db_path(db_path), read_only=False)
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
    db_path: Optional[str] = None,
) -> int:
    """선택한 ID 목록의 기사/기자 관점만 일부 삭제합니다."""
    clean_ids: List[int] = []
    for item in perspective_ids or []:
        if isinstance(item, bool):
            continue
        if isinstance(item, int) and item > 0:
            clean_ids.append(item)
        elif isinstance(item, str) and item.strip().isdigit() and int(item.strip()) > 0:
            clean_ids.append(int(item.strip()))

    if not clean_ids:
        return 0

    conn = get_sqlite_conn(db_path=_resolve_db_path(db_path), read_only=False)
    cur = conn.cursor()
    try:
        placeholders = ",".join(["?"] * len(clean_ids))
        cur.execute(f"DELETE FROM news_perspectives WHERE id IN ({placeholders});", clean_ids)
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


def delete_old_records_sqlite(
    older_than_iso: str,
    db_path: Optional[str] = None,
) -> int:
    """특정 기준 시각 이전의 오래된 데이터를 일부 삭제합니다."""
    cutoff = _normalize_dt_str(older_than_iso)
    conn = get_sqlite_conn(db_path=_resolve_db_path(db_path), read_only=False)
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM news_perspectives WHERE published_at < ?;", (cutoff,))
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
    db_path: Optional[str] = None,
    vacuum: bool = True,
    recreate_file: bool = False,
):
    """전체 데이터를 삭제합니다. recreate_file=True면 파일 자체를 지우고 스키마만 새로 만듭니다."""
    target_path = _resolve_db_path(db_path)
    if recreate_file:
        for suffix in ("", "-wal", "-shm"):
            target = f"{target_path}{suffix}"
            if os.path.exists(target):
                os.remove(target)
        init_sqlite_db(target_path)
        return

    if not os.path.exists(target_path):
        init_sqlite_db(target_path)
        return

    conn = get_sqlite_conn(db_path=target_path, read_only=False)
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
            conn.execute("VACUUM;")
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()