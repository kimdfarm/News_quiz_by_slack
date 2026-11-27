import json
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from psycopg2.extras import Json

from database import clean_old_news, get_db_connection

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_SQLITE_DIR = PROJECT_ROOT / "SQLITE"
DEFAULT_SQLITE_PATH = os.getenv(
    "NEWS_SQLITE_PATH",
    str(DEFAULT_SQLITE_DIR / "news_store.sqlite"),
)
STATE_FILE_PATH = PROJECT_ROOT / "storage_mode_state.json"




def normalize_storage_backend(backend: Optional[str]) -> str:
    val = (backend or "").strip().lower()
    if val in ("sqlite", "file", "local_file", "sq", "로컬", "파일"):
        return "sqlite"
    if val in ("postgres", "postgresql", "db", "pg", "neon", "neondb", "네온", "디비"):
        return "postgres"
    raise ValueError(
        f"지원하지 않는 STORAGE_BACKEND 값입니다: {backend} (사용 가능: postgres, sqlite)"
    )


def _read_state_file() -> Optional[str]:
    try:
        if STATE_FILE_PATH.exists():
            data = json.loads(STATE_FILE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("storage_mode"):
                return normalize_storage_backend(str(data["storage_mode"]))
    except Exception:
        pass
    return None


def _write_state_file(normalized: str) -> None:
    STATE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "storage_mode": normalized,
        "updated_at": datetime.now().isoformat(),
    }
    STATE_FILE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_active_storage_backend() -> str:
    from_file = _read_state_file()
    if from_file:
        return from_file

    current = (os.getenv("STORAGE_BACKEND") or "postgres").strip().lower()
    try:
        return normalize_storage_backend(current)
    except ValueError:
        return "postgres"


def set_active_storage_backend(backend: str) -> str:
    normalized = normalize_storage_backend(backend)
    _write_state_file(normalized)
    os.environ["STORAGE_BACKEND"] = normalized
    return normalized


def _resolve_sqlite_path(sqlite_path: Optional[str] = None) -> str:
    raw = (sqlite_path or os.getenv("NEWS_SQLITE_PATH") or DEFAULT_SQLITE_PATH).strip()
    if not raw:
        raw = str(DEFAULT_SQLITE_DIR / "news_store.sqlite")
    if not os.path.isabs(raw):
        raw = str(PROJECT_ROOT / raw)
    return os.path.abspath(raw)


def _parse_keywords_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(k).strip() for k in raw if str(k).strip()]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(k).strip() for k in parsed if str(k).strip()]
        except Exception:
            pass
        return [raw.strip()]
    return []


def _parse_json_obj(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=None) if raw.tzinfo else raw
    s = str(raw).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except Exception:
        return None


def _query_postgres_news(category: str, limit: int, days: int = 7) -> List[Dict[str, Any]]:
    clean_old_news()
    cutoff = datetime.now() - timedelta(days=days)

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT p.title, p.summary_facts, p.author_opinion, p.stance_type,
                   i.keywords, s.name AS source_name, p.published_at
            FROM news_perspectives p
            LEFT JOIN news_issues i ON i.id = p.issue_id
            LEFT JOIN categories c ON c.id = i.category_id
            LEFT JOIN sources s ON s.id = p.source_id
            WHERE (%s = '전체' OR c.name = %s)
              AND p.processing_status = 'COMPLETED'
              AND (p.published_at IS NULL OR p.published_at >= %s)
            ORDER BY p.id DESC
            LIMIT %s;
            """,
            (category, category, cutoff, limit),
        )
        rows = cur.fetchall()
        return [
            {
                "title": r[0],
                "facts": r[1] or "",
                "opinion": r[2] or "",
                "stance_type": r[3] or "중립",
                "keywords": _parse_keywords_list(r[4]),
                "source": r[5] or "",
                "published_at": r[6].isoformat() if isinstance(r[6], datetime) else (str(r[6]) if r[6] else ""),
            }
            for r in rows
        ]
    finally:
        cur.close()
        conn.close()


def _query_sqlite_news(
    category: str,
    limit: int,
    days: int = 7,
    db_path: str = DEFAULT_SQLITE_PATH,
) -> List[Dict[str, Any]]:
    resolved = _resolve_sqlite_path(db_path)
    if not os.path.exists(resolved):
        return []

    from crawler.sqlite_store import get_sqlite_conn

    cutoff_iso = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_sqlite_conn(db_path=resolved, read_only=True)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT p.title, p.summary_facts, p.author_opinion, p.stance_type,
                   i.keywords, s.name AS source_name, p.published_at
            FROM news_perspectives p
            LEFT JOIN news_issues i ON i.id = p.issue_id
            LEFT JOIN categories c ON c.id = i.category_id
            LEFT JOIN sources s ON s.id = p.source_id
            WHERE (? = '전체' OR c.name = ?)
              AND p.processing_status = 'COMPLETED'
              AND (p.published_at IS NULL OR p.published_at >= ?)
            ORDER BY p.id DESC
            LIMIT ?;
            """,
            (category, category, cutoff_iso, limit),
        )
        rows = cur.fetchall()
        return [
            {
                "title": r["title"],
                "facts": r["summary_facts"] or "",
                "opinion": r["author_opinion"] or "",
                "stance_type": r["stance_type"] or "중립",
                "keywords": _parse_keywords_list(r["keywords"]),
                "source": r["source_name"] or "",
                "published_at": r["published_at"] or "",
            }
            for r in rows
        ]
    finally:
        cur.close()
        conn.close()


def fetch_recent_news(
    category: str = "경제",
    limit: int = 3,
    days: int = 7,
    storage_backend: Optional[str] = None,
    sqlite_path: str = DEFAULT_SQLITE_PATH,
) -> Dict[str, Any]:
    active_backend = (
        set_active_storage_backend(storage_backend)
        if storage_backend
        else get_active_storage_backend()
    )
    resolved_path = _resolve_sqlite_path(sqlite_path)
    if active_backend == "sqlite":
        articles = _query_sqlite_news(category=category, limit=limit, days=days, db_path=resolved_path)
    else:
        articles = _query_postgres_news(category=category, limit=limit, days=days)

    return {
        "storage_backend": active_backend,
        "sqlite_path": resolved_path if active_backend == "sqlite" else None,
        "category": category,
        "count": len(articles),
        "articles": articles,
    }


def _ensure_sqlite_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    try:
        cur.executescript(
            """
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
            """
        )
        conn.commit()
    finally:
        cur.close()


def _get_or_create_sqlite_category(cur: sqlite3.Cursor, name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    cur.execute("INSERT OR IGNORE INTO categories (name) VALUES (?);", (name,))
    cur.execute("SELECT id FROM categories WHERE name = ?;", (name,))
    row = cur.fetchone()
    return int(row[0]) if row else None


def _get_or_create_sqlite_source(cur: sqlite3.Cursor, name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    cur.execute("INSERT OR IGNORE INTO sources (name) VALUES (?);", (name,))
    cur.execute("SELECT id FROM sources WHERE name = ?;", (name,))
    row = cur.fetchone()
    return int(row[0]) if row else None


def _get_or_create_sqlite_issue(
    cur: sqlite3.Cursor, issue_row: Optional[Dict[str, Any]], category_id: Optional[int]
) -> Optional[int]:
    if not issue_row:
        return None
    issue_hash = issue_row.get("issue_hash")
    if issue_hash:
        cur.execute("SELECT id FROM news_issues WHERE issue_hash = ?;", (issue_hash,))
        existing = cur.fetchone()
        if existing:
            return int(existing[0])

    keywords_json = json.dumps(issue_row.get("keywords") or [], ensure_ascii=False)
    cur.execute(
        """
        INSERT INTO news_issues (
            issue_title, representative_summary, category_id, keywords,
            issue_hash, published_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            issue_row.get("issue_title") or "",
            issue_row.get("representative_summary"),
            category_id,
            keywords_json,
            issue_hash,
            issue_row.get("published_at"),
            issue_row.get("created_at"),
            issue_row.get("updated_at"),
        ),
    )
    return int(cur.lastrowid)


def _get_or_create_pg_category(cur: Any, name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    cur.execute(
        """
        INSERT INTO categories (name) VALUES (%s)
        ON CONFLICT (name) DO NOTHING;
        """,
        (name,),
    )
    cur.execute("SELECT id FROM categories WHERE name = %s;", (name,))
    row = cur.fetchone()
    return int(row[0]) if row else None


def _get_or_create_pg_source(cur: Any, name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    cur.execute(
        """
        INSERT INTO sources (name) VALUES (%s)
        ON CONFLICT (name) DO NOTHING;
        """,
        (name,),
    )
    cur.execute("SELECT id FROM sources WHERE name = %s;", (name,))
    row = cur.fetchone()
    return int(row[0]) if row else None


def _get_or_create_pg_issue(
    cur: Any, issue_row: Optional[Dict[str, Any]], category_id: Optional[int]
) -> Optional[int]:
    if not issue_row:
        return None
    issue_hash = issue_row.get("issue_hash")
    if issue_hash:
        cur.execute("SELECT id FROM news_issues WHERE issue_hash = %s;", (issue_hash,))
        existing = cur.fetchone()
        if existing:
            return int(existing[0])

    keywords = _parse_keywords_list(issue_row.get("keywords"))
    cur.execute(
        """
        INSERT INTO news_issues (
            issue_title, representative_summary, category_id, keywords,
            issue_hash, published_at, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s, CURRENT_TIMESTAMP), COALESCE(%s, CURRENT_TIMESTAMP))
        RETURNING id;
        """,
        (
            issue_row.get("issue_title") or "",
            issue_row.get("representative_summary"),
            category_id,
            keywords,
            issue_hash,
            _parse_timestamp(issue_row.get("published_at")),
            _parse_timestamp(issue_row.get("created_at")),
            _parse_timestamp(issue_row.get("updated_at")),
        ),
    )
    row = cur.fetchone()
    return int(row[0]) if row else None


def sync_sqlite_to_postgres(
    limit: Optional[int] = None,
    sqlite_path: Optional[str] = None,
) -> Dict[str, Any]:
    """로컬 SQLite -> Neon DB(PostgreSQL) 단방향 동기화 (중복 스킵, 원본 유지)."""
    db_file = _resolve_sqlite_path(sqlite_path)
    if not os.path.isfile(db_file):
        return {
            "status": "unavailable",
            "direction": "sqlite_to_postgres",
            "limit": limit,
            "message": f"SQLite 파일이 없어요: {db_file}",
        }

    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if not db_url:
        return {
            "status": "unavailable",
            "direction": "sqlite_to_postgres",
            "limit": limit,
            "message": "DATABASE_URL 환경변수가 설정되지 않았어요.",
        }

    sq_conn = sqlite3.connect(db_file)
    sq_conn.row_factory = sqlite3.Row
    sq_cur = sq_conn.cursor()

    pg_conn = get_db_connection()
    pg_cur = pg_conn.cursor()

    try:
        sql = """
            SELECT
                p.*,
                s.name AS source_name,
                i.issue_title,
                i.representative_summary,
                i.keywords,
                i.issue_hash,
                i.published_at AS issue_published_at,
                i.created_at AS issue_created_at,
                i.updated_at AS issue_updated_at,
                c.name AS category_name
            FROM news_perspectives p
            LEFT JOIN sources s ON p.source_id = s.id
            LEFT JOIN news_issues i ON p.issue_id = i.id
            LEFT JOIN categories c ON i.category_id = c.id
            ORDER BY COALESCE(p.published_at, p.created_at) DESC, p.id DESC
        """
        params: List[Any] = []
        if limit is not None and int(limit) > 0:
            sql += " LIMIT ?"
            params.append(int(limit))
        sql += ";"

        sq_cur.execute(sql, params)
        rows = [dict(r) for r in sq_cur.fetchall()]

        inserted = 0
        skipped = 0
        seen_urls_in_batch: Set[str] = set()
        seen_hashes_in_batch: Set[str] = set()

        for r in rows:
            raw_url = (r.get("url") or "").strip()
            norm_url = raw_url.lower()
            content_hash = (r.get("content_hash") or "").strip()
            title = (r.get("title") or "").strip()

            # 1) URL 기준 중복 체크 (현재 배치 + 기존 Neon DB)
            if norm_url:
                if norm_url in seen_urls_in_batch:
                    skipped += 1
                    continue
                pg_cur.execute(
                    "SELECT 1 FROM news_perspectives WHERE LOWER(TRIM(url)) = %s LIMIT 1;",
                    (norm_url,),
                )
                if pg_cur.fetchone():
                    seen_urls_in_batch.add(norm_url)
                    skipped += 1
                    continue

            # 2) content_hash 기준 중복 체크
            if content_hash:
                if content_hash in seen_hashes_in_batch:
                    skipped += 1
                    continue
                pg_cur.execute(
                    "SELECT 1 FROM news_perspectives WHERE content_hash = %s LIMIT 1;",
                    (content_hash,),
                )
                if pg_cur.fetchone():
                    seen_hashes_in_batch.add(content_hash)
                    skipped += 1
                    continue

            # 3) URL도 해시도 없을 때 제목 기준 중복 체크
            if not norm_url and not content_hash and title:
                pg_cur.execute(
                    "SELECT 1 FROM news_perspectives WHERE TRIM(title) = %s LIMIT 1;",
                    (title,),
                )
                if pg_cur.fetchone():
                    skipped += 1
                    continue

            if not norm_url and not content_hash and not title:
                skipped += 1
                continue

            cat_id = _get_or_create_pg_category(pg_cur, r.get("category_name"))
            src_id = _get_or_create_pg_source(pg_cur, r.get("source_name"))
            issue_data = (
                {
                    "issue_title": r.get("issue_title"),
                    "representative_summary": r.get("representative_summary"),
                    "keywords": _parse_keywords_list(r.get("keywords")),
                    "issue_hash": r.get("issue_hash"),
                    "published_at": r.get("issue_published_at"),
                    "created_at": r.get("issue_created_at"),
                    "updated_at": r.get("issue_updated_at"),
                }
                if r.get("issue_title") or r.get("issue_hash")
                else None
            )
            issue_id = _get_or_create_pg_issue(pg_cur, issue_data, cat_id)
            meta_obj = _parse_json_obj(r.get("metadata"))

            pg_cur.execute(
                """
                INSERT INTO news_perspectives (
                    issue_id, source_id, title, author_name, url,
                    summary_facts, author_opinion, stance_type, raw_content,
                    content_hash, thumbnail_url, processing_status,
                    model_version, metadata, published_at, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, COALESCE(%s, CURRENT_TIMESTAMP)
                )
                ON CONFLICT (url) DO NOTHING;
                """,
                (
                    issue_id,
                    src_id,
                    title,
                    r.get("author_name"),
                    raw_url or None,
                    r.get("summary_facts"),
                    r.get("author_opinion"),
                    r.get("stance_type") or "중립",
                    r.get("raw_content"),
                    content_hash or None,
                    r.get("thumbnail_url"),
                    r.get("processing_status") or "COMPLETED",
                    r.get("model_version"),
                    Json(meta_obj),
                    _parse_timestamp(r.get("published_at")),
                    _parse_timestamp(r.get("created_at")),
                ),
            )
            if pg_cur.rowcount and pg_cur.rowcount > 0:
                inserted += 1
                if norm_url:
                    seen_urls_in_batch.add(norm_url)
                if content_hash:
                    seen_hashes_in_batch.add(content_hash)
            else:
                skipped += 1

        pg_conn.commit()
        return {
            "status": "ok",
            "direction": "sqlite_to_postgres",
            "limit": limit,
            "fetched_count": len(rows),
            "inserted_count": inserted,
            "skipped_count": skipped,
        }
    except Exception as e:
        pg_conn.rollback()
        return {
            "status": "error",
            "direction": "sqlite_to_postgres",
            "limit": limit,
            "message": f"SQLite -> Neon DB 동기화 실패: {e}",
        }
    finally:
        sq_cur.close()
        sq_conn.close()
        pg_cur.close()
        pg_conn.close()


def sync_postgres_to_sqlite(
    limit: Optional[int] = None,
    sqlite_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Neon DB(PostgreSQL) -> 로컬 SQLite 단방향 동기화 (중복 스킵, 원본 유지)."""
    db_file = _resolve_sqlite_path(sqlite_path)
    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if not db_url:
        return {
            "status": "unavailable",
            "direction": "postgres_to_sqlite",
            "limit": limit,
            "message": "DATABASE_URL 환경변수가 설정되지 않았어요.",
        }

    os.makedirs(os.path.dirname(db_file) or ".", exist_ok=True)
    sq_conn = sqlite3.connect(db_file)
    _ensure_sqlite_tables(sq_conn)
    sq_cur = sq_conn.cursor()

    pg_conn = get_db_connection()
    pg_cur = pg_conn.cursor()

    try:
        sql = """
            SELECT
                p.issue_id,
                p.source_id,
                p.title,
                p.author_name,
                p.url,
                p.summary_facts,
                p.author_opinion,
                p.stance_type,
                p.raw_content,
                p.content_hash,
                p.thumbnail_url,
                p.processing_status,
                p.model_version,
                p.metadata,
                p.published_at,
                p.created_at,
                s.name AS source_name,
                i.issue_title,
                i.representative_summary,
                i.keywords,
                i.issue_hash,
                i.published_at AS issue_published_at,
                i.created_at AS issue_created_at,
                i.updated_at AS issue_updated_at,
                c.name AS category_name
            FROM news_perspectives p
            LEFT JOIN sources s ON p.source_id = s.id
            LEFT JOIN news_issues i ON p.issue_id = i.id
            LEFT JOIN categories c ON i.category_id = c.id
            ORDER BY COALESCE(p.published_at, p.created_at) DESC, p.id DESC
        """
        params: List[Any] = []
        if limit is not None and int(limit) > 0:
            sql += " LIMIT %s"
            params.append(int(limit))
        sql += ";"

        pg_cur.execute(sql, params)
        cols = [desc[0] for desc in pg_cur.description]
        rows = [dict(zip(cols, r)) for r in pg_cur.fetchall()]

        inserted = 0
        skipped = 0
        seen_urls_in_batch: Set[str] = set()
        seen_hashes_in_batch: Set[str] = set()

        for r in rows:
            raw_url = (r.get("url") or "").strip()
            norm_url = raw_url.lower()
            content_hash = (r.get("content_hash") or "").strip()
            title = (r.get("title") or "").strip()

            # 1) URL 기준 중복 체크 (현재 배치 + 기존 SQLite DB)
            if norm_url:
                if norm_url in seen_urls_in_batch:
                    skipped += 1
                    continue
                sq_cur.execute(
                    "SELECT 1 FROM news_perspectives WHERE LOWER(TRIM(url)) = ? LIMIT 1;",
                    (norm_url,),
                )
                if sq_cur.fetchone():
                    seen_urls_in_batch.add(norm_url)
                    skipped += 1
                    continue

            # 2) content_hash 기준 중복 체크
            if content_hash:
                if content_hash in seen_hashes_in_batch:
                    skipped += 1
                    continue
                sq_cur.execute(
                    "SELECT 1 FROM news_perspectives WHERE content_hash = ? LIMIT 1;",
                    (content_hash,),
                )
                if sq_cur.fetchone():
                    seen_hashes_in_batch.add(content_hash)
                    skipped += 1
                    continue

            # 3) URL도 해시도 없을 때 제목 기준 중복 체크
            if not norm_url and not content_hash and title:
                sq_cur.execute(
                    "SELECT 1 FROM news_perspectives WHERE TRIM(title) = ? LIMIT 1;",
                    (title,),
                )
                if sq_cur.fetchone():
                    skipped += 1
                    continue

            if not norm_url and not content_hash and not title:
                skipped += 1
                continue

            cat_id = _get_or_create_sqlite_category(sq_cur, r.get("category_name"))
            src_id = _get_or_create_sqlite_source(sq_cur, r.get("source_name"))
            issue_data = (
                {
                    "issue_title": r.get("issue_title"),
                    "representative_summary": r.get("representative_summary"),
                    "keywords": _parse_keywords_list(r.get("keywords")),
                    "issue_hash": r.get("issue_hash"),
                    "published_at": r.get("issue_published_at"),
                    "created_at": r.get("issue_created_at"),
                    "updated_at": r.get("issue_updated_at"),
                }
                if r.get("issue_title") or r.get("issue_hash")
                else None
            )
            issue_id = _get_or_create_sqlite_issue(sq_cur, issue_data, cat_id)
            meta_json = json.dumps(_parse_json_obj(r.get("metadata")), ensure_ascii=False)

            pub_at = r.get("published_at")
            pub_at_str = pub_at.isoformat() if isinstance(pub_at, datetime) else (str(pub_at) if pub_at else None)
            crt_at = r.get("created_at")
            crt_at_str = crt_at.isoformat() if isinstance(crt_at, datetime) else (str(crt_at) if crt_at else None)

            sq_cur.execute(
                """
                INSERT OR IGNORE INTO news_perspectives (
                    issue_id, source_id, title, author_name, url,
                    summary_facts, author_opinion, stance_type, raw_content,
                    content_hash, thumbnail_url, processing_status,
                    model_version, metadata, published_at, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                );
                """,
                (
                    issue_id,
                    src_id,
                    title,
                    r.get("author_name"),
                    raw_url or None,
                    r.get("summary_facts"),
                    r.get("author_opinion"),
                    r.get("stance_type") or "중립",
                    r.get("raw_content"),
                    content_hash or None,
                    r.get("thumbnail_url"),
                    r.get("processing_status") or "COMPLETED",
                    r.get("model_version"),
                    meta_json,
                    pub_at_str,
                    crt_at_str,
                ),
            )
            if sq_cur.rowcount and sq_cur.rowcount > 0:
                inserted += 1
                if norm_url:
                    seen_urls_in_batch.add(norm_url)
                if content_hash:
                    seen_hashes_in_batch.add(content_hash)
            else:
                skipped += 1

        sq_conn.commit()
        sq_cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        sq_conn.commit()
        return {
            "status": "ok",
            "direction": "postgres_to_sqlite",
            "limit": limit,
            "fetched_count": len(rows),
            "inserted_count": inserted,
            "skipped_count": skipped,
            "sqlite_path": db_file,
        }
    except Exception as e:
        sq_conn.rollback()
        return {
            "status": "error",
            "direction": "postgres_to_sqlite",
            "limit": limit,
            "message": f"Neon DB -> SQLite 동기화 실패: {e}",
        }
    finally:
        sq_cur.close()
        sq_conn.close()
        pg_cur.close()
        pg_conn.close()

def _known_press_prefixes() -> Set[str]:
    names: Set[str] = set()
    try:
        from crawler.news.rss_sources import load_press_rss_map

        for key in load_press_rss_map().keys():
            clean_key = str(key).strip()
            if not clean_key:
                continue
            names.add(clean_key)
            prefix = clean_key.split("/", 1)[0].strip()
            if prefix:
                names.add(prefix)
    except Exception:
        pass
    return names


def _expand_source_candidates(raw_source: str, known_names: Set[str]) -> List[str]:
    s = re.sub(r"\s+", " ", (raw_source or "").strip())
    if not s:
        return []

    candidates: Set[str] = {s}
    stripped_suffix = re.sub(r"(일보|신문|방송|뉴스|통신)$", "", s).strip()
    if stripped_suffix:
        candidates.add(stripped_suffix)

    matched: Set[str] = set()
    for cand in list(candidates):
        if cand in known_names:
            matched.add(cand)
        for known in known_names:
            if known == cand or known.startswith(f"{cand}/"):
                matched.add(known)

    return sorted(matched if matched else candidates)




REQUIRED_SQLITE_SCHEMA: Dict[str, Set[str]] = {
    "categories": {"id", "name"},
    "sources": {"id", "name"},
    "news_issues": {"id", "issue_title", "category_id", "keywords", "issue_hash"},
    "news_perspectives": {
        "id",
        "issue_id",
        "source_id",
        "title",
        "url",
        "summary_facts",
        "author_opinion",
        "stance_type",
        "processing_status",
        "published_at",
        "created_at",
    },
}


def validate_sqlite_file_schema(candidate_path: str) -> Dict[str, Any]:
    """파일 이름과 무관하게 SQLite 헤더와 필수 테이블/컬럼 구조가 일치하는지 검증합니다."""
    if not os.path.isfile(candidate_path):
        return {"valid": False, "reason": "업로드된 파일을 찾을 수 없어요."}

    try:
        with open(candidate_path, "rb") as f:
            header = f.read(16)
        if header != b"SQLite format 3\x00":
            return {"valid": False, "reason": "유효한 SQLite 3 데이터베이스 파일이 아니에요."}
    except Exception as e:
        return {"valid": False, "reason": f"파일 헤더를 읽을 수 없어요: {e}"}

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(f"file:{os.path.abspath(candidate_path)}?mode=ro", uri=True)
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = {str(r[0]) for r in cur.fetchall()}

        missing_tables = sorted(set(REQUIRED_SQLITE_SCHEMA.keys()) - existing_tables)
        if missing_tables:
            return {
                "valid": False,
                "reason": f"필수 테이블이 없어 구조가 맞지 않아요: {', '.join(missing_tables)}",
            }

        for table_name, required_cols in REQUIRED_SQLITE_SCHEMA.items():
            cur.execute(f"PRAGMA table_info({table_name});")
            actual_cols = {str(r[1]) for r in cur.fetchall()}
            missing_cols = sorted(required_cols - actual_cols)
            if missing_cols:
                return {
                    "valid": False,
                    "reason": f"`{table_name}` 테이블에 필수 컬럼이 없어요: {', '.join(missing_cols)}",
                }

        cur.execute("SELECT COUNT(*) FROM news_perspectives;")
        row_count = int(cur.fetchone()[0] or 0)
        return {"valid": True, "row_count": row_count}
    except sqlite3.DatabaseError as e:
        return {"valid": False, "reason": f"손상되었거나 열 수 없는 SQLite 파일이에요: {e}"}
    finally:
        if conn is not None:
            conn.close()


def import_sqlite_file_data(
    uploaded_sqlite_path: str,
    target_sqlite_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    업로드된 SQLite 파일의 스키마를 먼저 검증한 뒤,
    구조가 일치할 때만 대상 SQLite DB에 데이터를 병합합니다(중복 URL 스킵).
    """
    check = validate_sqlite_file_schema(uploaded_sqlite_path)
    if not check.get("valid"):
        return {
            "status": "rejected",
            "inserted_count": 0,
            "skipped_count": 0,
            "message": check.get("reason") or "파일 구조가 일치하지 않아 반영하지 않았어요.",
        }

    target_file = _resolve_sqlite_path(target_sqlite_path)
    os.makedirs(os.path.dirname(target_file) or ".", exist_ok=True)

    src_conn = sqlite3.connect(f"file:{os.path.abspath(uploaded_sqlite_path)}?mode=ro", uri=True)
    src_conn.row_factory = sqlite3.Row
    src_cur = src_conn.cursor()

    dst_conn = sqlite3.connect(target_file, timeout=15.0)
    dst_conn.execute("PRAGMA journal_mode=WAL;")
    dst_conn.execute("PRAGMA busy_timeout=5000;")
    dst_conn.execute("PRAGMA foreign_keys=ON;")
    _ensure_sqlite_tables(dst_conn)
    dst_cur = dst_conn.cursor()

    try:
        src_cur.execute(
            """
            SELECT
                p.*,
                s.name AS source_name,
                i.issue_title,
                i.representative_summary,
                i.keywords,
                i.issue_hash,
                i.published_at AS issue_published_at,
                i.created_at AS issue_created_at,
                i.updated_at AS issue_updated_at,
                c.name AS category_name
            FROM news_perspectives p
            LEFT JOIN sources s ON p.source_id = s.id
            LEFT JOIN news_issues i ON p.issue_id = i.id
            LEFT JOIN categories c ON i.category_id = c.id
            ORDER BY p.id ASC;
            """
        )
        rows = [dict(r) for r in src_cur.fetchall()]

        inserted = 0
        skipped = 0

        for r in rows:
            url = (r.get("url") or "").strip()
            if not url:
                skipped += 1
                continue

            dst_cur.execute("SELECT 1 FROM news_perspectives WHERE url = ?;", (url,))
            if dst_cur.fetchone():
                skipped += 1
                continue

            cat_id = _get_or_create_sqlite_category(dst_cur, r.get("category_name"))
            src_id = _get_or_create_sqlite_source(dst_cur, r.get("source_name"))
            issue_data = (
                {
                    "issue_title": r.get("issue_title"),
                    "representative_summary": r.get("representative_summary"),
                    "keywords": _parse_keywords_list(r.get("keywords")),
                    "issue_hash": r.get("issue_hash"),
                    "published_at": r.get("issue_published_at"),
                    "created_at": r.get("issue_created_at"),
                    "updated_at": r.get("issue_updated_at"),
                }
                if r.get("issue_title") or r.get("issue_hash")
                else None
            )
            issue_id = _get_or_create_sqlite_issue(dst_cur, issue_data, cat_id)
            meta_json = json.dumps(_parse_json_obj(r.get("metadata")), ensure_ascii=False)

            dst_cur.execute(
                """
                INSERT OR IGNORE INTO news_perspectives (
                    issue_id, source_id, title, author_name, url,
                    summary_facts, author_opinion, stance_type, raw_content,
                    content_hash, thumbnail_url, processing_status,
                    model_version, metadata, published_at, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                );
                """,
                (
                    issue_id,
                    src_id,
                    r.get("title") or "",
                    r.get("author_name"),
                    url,
                    r.get("summary_facts"),
                    r.get("author_opinion"),
                    r.get("stance_type") or "중립",
                    r.get("raw_content"),
                    r.get("content_hash"),
                    r.get("thumbnail_url"),
                    r.get("processing_status") or "COMPLETED",
                    r.get("model_version"),
                    meta_json,
                    r.get("published_at"),
                    r.get("created_at"),
                ),
            )
            if dst_cur.rowcount and dst_cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

        dst_conn.commit()
        dst_cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        dst_conn.commit()

        return {
            "status": "ok",
            "fetched_count": len(rows),
            "inserted_count": inserted,
            "skipped_count": skipped,
            "sqlite_path": target_file,
        }
    except Exception as e:
        dst_conn.rollback()
        return {
            "status": "error",
            "inserted_count": 0,
            "skipped_count": 0,
            "message": f"SQLite 데이터 병합 중 오류가 발생해 롤백했어요: {e}",
        }
    finally:
        src_cur.close()
        src_conn.close()
        dst_cur.close()
        dst_conn.close()
        
        

def read_persisted_storage_mode(default: str = "postgres") -> str:
    try:
        if STATE_FILE_PATH.exists():
            payload = json.loads(STATE_FILE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("storage_mode"):
                return normalize_storage_backend(str(payload["storage_mode"]))
    except Exception:
        pass

    env_mode = (os.getenv("STORAGE_BACKEND") or "").strip()
    if env_mode:
        try:
            return normalize_storage_backend(env_mode)
        except ValueError:
            pass
    return normalize_storage_backend(default)


def _resolve_target_backend_from_state(target_backend: Optional[str]) -> str:
    persisted = read_persisted_storage_mode()
    if target_backend is None:
        return persisted
    raw = str(target_backend).strip()
    if not raw or "|" in raw or raw.lower() in ("null", "none", "auto", "default"):
        return persisted
    try:
        return normalize_storage_backend(raw)
    except ValueError:
        return persisted


def _import_sqlite_delete_helpers():
    try:
        from crawler.sqlite_store import (
            clear_all_sqlite_data,
            delete_old_records_sqlite,
        )
        return clear_all_sqlite_data, delete_old_records_sqlite
    except Exception:
        from sqlite_store import (
            clear_all_sqlite_data,
            delete_old_records_sqlite,
        )
        return clear_all_sqlite_data, delete_old_records_sqlite


def _delete_from_sqlite(
    db_file: str,
    effective_delete_all: bool,
    clean_source: Optional[str],
    source_candidates: List[str],
    clean_category: Optional[str],
    clean_keyword: Optional[str],
    clean_days: Optional[int],
    clean_limit: Optional[int],
    order: str,
    applied_filters: Dict[str, Any],
) -> Dict[str, Any]:
    if not os.path.isfile(db_file):
        return {
            "status": "unavailable",
            "backend": "sqlite",
            "deleted_count": 0,
            "filters": applied_filters,
            "message": f"SQLite 파일이 없어요: {db_file}",
        }

    clear_all_sqlite_data, delete_old_records_sqlite = _import_sqlite_delete_helpers()

    try:
        # 1) 전체 삭제
        if effective_delete_all:
            conn = sqlite3.connect(db_file, timeout=15.0)
            cur = conn.cursor()
            try:
                cur.execute("SELECT COUNT(*) FROM news_perspectives;")
                before_count = int((cur.fetchone() or [0])[0] or 0)
            finally:
                cur.close()
                conn.close()

            clear_all_sqlite_data(db_path=db_file, vacuum=True, recreate_file=False)
            return {
                "status": "ok",
                "backend": "sqlite",
                "deleted_count": before_count,
                "filters": applied_filters,
                "sqlite_path": db_file,
            }

        # 2) 기간(days) 단독 삭제
        if (
            clean_days is not None
            and not clean_source
            and not clean_category
            and not clean_keyword
            and clean_limit is None
        ):
            cutoff_iso = (datetime.now() - timedelta(days=clean_days)).strftime("%Y-%m-%d %H:%M:%S")
            deleted_count = delete_old_records_sqlite(older_than_iso=cutoff_iso, db_path=db_file)
            return {
                "status": "ok",
                "backend": "sqlite",
                "deleted_count": int(deleted_count or 0),
                "filters": applied_filters,
                "sqlite_path": db_file,
            }

        # 3) 복합 조건 / 건수 제한 삭제
        conn = sqlite3.connect(db_file, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        cur = conn.cursor()
        try:
            where_clauses: List[str] = []
            params: List[Any] = []

            if clean_source:
                placeholders = ",".join(["?"] * len(source_candidates))
                like_clauses = " OR ".join(["name LIKE ?"] * len(source_candidates))
                like_params = [f"{c}/%" for c in source_candidates]
                cur.execute(
                    f"""
                    SELECT id, name
                    FROM sources
                    WHERE name IN ({placeholders}) OR {like_clauses};
                    """,
                    source_candidates + like_params,
                )
                matched_source_ids = [int(r[0]) for r in cur.fetchall()]
                if not matched_source_ids:
                    return {
                        "status": "ok",
                        "backend": "sqlite",
                        "deleted_count": 0,
                        "filters": applied_filters,
                        "sqlite_path": db_file,
                        "message": f"'{clean_source}'에 해당하는 저장된 언론사가 없어 삭제하지 않았어요.",
                    }
                src_ph = ",".join(["?"] * len(matched_source_ids))
                where_clauses.append(f"p.source_id IN ({src_ph})")
                params.extend(matched_source_ids)

            if clean_category:
                where_clauses.append("c.name = ?")
                params.append(clean_category)

            if clean_keyword:
                kw_pat = f"%{clean_keyword}%"
                where_clauses.append(
                    "(p.title LIKE ? OR COALESCE(p.summary_facts, '') LIKE ? "
                    "OR COALESCE(p.author_opinion, '') LIKE ? OR COALESCE(i.keywords, '') LIKE ?)"
                )
                params.extend([kw_pat, kw_pat, kw_pat, kw_pat])

            if clean_days is not None:
                cutoff_str = (datetime.now() - timedelta(days=clean_days)).strftime("%Y-%m-%d %H:%M:%S")
                where_clauses.append("COALESCE(p.published_at, p.created_at) >= ?")
                params.append(cutoff_str)

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            order_dir = "ASC" if str(order).strip().lower() == "oldest" else "DESC"

            if clean_limit is not None:
                select_sql = f"""
                    SELECT p.id
                    FROM news_perspectives p
                    LEFT JOIN sources s ON s.id = p.source_id
                    LEFT JOIN news_issues i ON i.id = p.issue_id
                    LEFT JOIN categories c ON c.id = i.category_id
                    {where_sql}
                    ORDER BY COALESCE(p.published_at, p.created_at) {order_dir}, p.id {order_dir}
                    LIMIT ?;
                """
                cur.execute(select_sql, params + [clean_limit])
                target_ids = [int(r[0]) for r in cur.fetchall()]
                if target_ids:
                    placeholders = ",".join(["?"] * len(target_ids))
                    cur.execute(f"DELETE FROM news_perspectives WHERE id IN ({placeholders});", target_ids)
                    deleted_count = cur.rowcount or 0
                else:
                    deleted_count = 0
            else:
                delete_sql = f"""
                    DELETE FROM news_perspectives
                    WHERE id IN (
                        SELECT p.id
                        FROM news_perspectives p
                        LEFT JOIN sources s ON s.id = p.source_id
                        LEFT JOIN news_issues i ON i.id = p.issue_id
                        LEFT JOIN categories c ON c.id = i.category_id
                        {where_sql}
                    );
                """
                cur.execute(delete_sql, params)
                deleted_count = cur.rowcount or 0

            cur.execute(
                """
                DELETE FROM news_issues
                WHERE id NOT IN (
                    SELECT DISTINCT issue_id FROM news_perspectives WHERE issue_id IS NOT NULL
                );
                """
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

        ckpt_conn = sqlite3.connect(db_file, timeout=15.0, isolation_level=None)
        try:
            ckpt_conn.execute("PRAGMA busy_timeout=5000;")
            ckpt_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            if int(deleted_count) > 0:
                ckpt_conn.execute("VACUUM;")
        except sqlite3.OperationalError:
            pass
        finally:
            ckpt_conn.close()

        return {
            "status": "ok",
            "backend": "sqlite",
            "deleted_count": int(deleted_count),
            "filters": applied_filters,
            "sqlite_path": db_file,
        }
    except Exception as e:
        return {
            "status": "error",
            "backend": "sqlite",
            "deleted_count": 0,
            "filters": applied_filters,
            "message": f"SQLite 삭제 실패: {e}",
        }


def _delete_from_postgres(
    effective_delete_all: bool,
    clean_source: Optional[str],
    source_candidates: List[str],
    clean_category: Optional[str],
    clean_keyword: Optional[str],
    clean_days: Optional[int],
    clean_limit: Optional[int],
    order: str,
    applied_filters: Dict[str, Any],
) -> Dict[str, Any]:
    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if not db_url:
        return {
            "status": "unavailable",
            "backend": "postgres",
            "deleted_count": 0,
            "filters": applied_filters,
            "message": "DATABASE_URL 환경변수가 설정되지 않았어요.",
        }

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            if effective_delete_all:
                cur.execute("SELECT COUNT(*) FROM news_perspectives;")
                before_count = int((cur.fetchone() or [0])[0] or 0)

                cur.execute("DELETE FROM news_perspectives;")
                cur.execute(
                    """
                    DELETE FROM news_issues
                    WHERE id NOT IN (
                        SELECT DISTINCT issue_id FROM news_perspectives WHERE issue_id IS NOT NULL
                    );
                    """
                )
                conn.commit()
                return {
                    "status": "ok",
                    "backend": "postgres",
                    "deleted_count": before_count,
                    "filters": applied_filters,
                }

            where_clauses: List[str] = []
            params: List[Any] = []

            if clean_source:
                placeholders = ",".join(["%s"] * len(source_candidates))
                like_clauses = " OR ".join(["name LIKE %s"] * len(source_candidates))
                like_params = [f"{c}/%" for c in source_candidates]
                cur.execute(
                    f"""
                    SELECT id
                    FROM sources
                    WHERE name IN ({placeholders}) OR {like_clauses};
                    """,
                    source_candidates + like_params,
                )
                matched_source_ids = [int(r[0]) for r in cur.fetchall()]
                if not matched_source_ids:
                    return {
                        "status": "ok",
                        "backend": "postgres",
                        "deleted_count": 0,
                        "filters": applied_filters,
                        "message": f"'{clean_source}'에 해당하는 저장된 언론사가 없어 삭제하지 않았어요.",
                    }
                src_ph = ",".join(["%s"] * len(matched_source_ids))
                where_clauses.append(f"p.source_id IN ({src_ph})")
                params.extend(matched_source_ids)

            if clean_category:
                where_clauses.append("c.name = %s")
                params.append(clean_category)

            if clean_keyword:
                kw_pat = f"%{clean_keyword}%"
                where_clauses.append(
                    "(p.title LIKE %s OR COALESCE(p.summary_facts, '') LIKE %s "
                    "OR COALESCE(p.author_opinion, '') LIKE %s OR COALESCE(i.keywords::text, '') LIKE %s)"
                )
                params.extend([kw_pat, kw_pat, kw_pat, kw_pat])

            if clean_days is not None:
                cutoff_dt = datetime.now() - timedelta(days=clean_days)
                where_clauses.append("COALESCE(p.published_at, p.created_at) >= %s")
                params.append(cutoff_dt)

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            order_dir = "ASC" if str(order).strip().lower() == "oldest" else "DESC"

            if clean_limit is not None:
                select_sql = f"""
                    SELECT p.id
                    FROM news_perspectives p
                    LEFT JOIN sources s ON s.id = p.source_id
                    LEFT JOIN news_issues i ON i.id = p.issue_id
                    LEFT JOIN categories c ON c.id = i.category_id
                    {where_sql}
                    ORDER BY COALESCE(p.published_at, p.created_at) {order_dir}, p.id {order_dir}
                    LIMIT %s;
                """
                cur.execute(select_sql, params + [clean_limit])
                target_ids = [int(r[0]) for r in cur.fetchall()]
                if target_ids:
                    placeholders = ",".join(["%s"] * len(target_ids))
                    cur.execute(f"DELETE FROM news_perspectives WHERE id IN ({placeholders});", target_ids)
                    deleted_count = cur.rowcount or 0
                else:
                    deleted_count = 0
            else:
                delete_sql = f"""
                    DELETE FROM news_perspectives
                    WHERE id IN (
                        SELECT p.id
                        FROM news_perspectives p
                        LEFT JOIN sources s ON s.id = p.source_id
                        LEFT JOIN news_issues i ON i.id = p.issue_id
                        LEFT JOIN categories c ON c.id = i.category_id
                        {where_sql}
                    );
                """
                cur.execute(delete_sql, params)
                deleted_count = cur.rowcount or 0

            cur.execute(
                """
                DELETE FROM news_issues
                WHERE id NOT IN (
                    SELECT DISTINCT issue_id FROM news_perspectives WHERE issue_id IS NOT NULL
                );
                """
            )
            conn.commit()
            return {
                "status": "ok",
                "backend": "postgres",
                "deleted_count": int(deleted_count),
                "filters": applied_filters,
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        return {
            "status": "error",
            "backend": "postgres",
            "deleted_count": 0,
            "filters": applied_filters,
            "message": f"Neon DB 삭제 실패: {e}",
        }


def delete_news_by_filter(
    source: Optional[str] = None,
    category: Optional[str] = None,
    keyword: Optional[str] = None,
    days: Optional[int] = None,
    limit: Optional[int] = None,
    order: str = "newest",
    delete_all: bool = False,
    target_backend: Optional[str] = None,
    sqlite_path: Optional[str] = None,
) -> Dict[str, Any]:
    chosen_backend = _resolve_target_backend_from_state(target_backend)
    want_delete_all = bool(delete_all)

    clean_source = None if want_delete_all else ((source or "").strip() or None)
    clean_category = None if want_delete_all else ((category or "").strip() or None)
    clean_keyword = None if want_delete_all else ((keyword or "").strip() or None)
    clean_days = None if want_delete_all else (int(days) if days is not None and int(days) > 0 else None)
    clean_limit = None if want_delete_all else (int(limit) if limit is not None and int(limit) > 0 else None)

    has_any_filter = any([
        bool(clean_source),
        bool(clean_category),
        bool(clean_keyword),
        clean_days is not None,
        clean_limit is not None,
    ])
    effective_delete_all = want_delete_all and not has_any_filter

    if not effective_delete_all and not has_any_filter:
        return {
            "status": "rejected",
            "backend": chosen_backend,
            "deleted_count": 0,
            "message": "삭제 조건(언론사, 분야, 키워드, 기간, 건수)이 없어요. 조건을 지정해 주세요.",
        }

    known_names = _known_press_prefixes()
    source_candidates = _expand_source_candidates(clean_source, known_names) if clean_source else []

    applied_filters = {
        "source": clean_source,
        "matched_sources": source_candidates if clean_source else [],
        "category": clean_category,
        "keyword": clean_keyword,
        "days": clean_days,
        "limit": clean_limit,
        "delete_all": effective_delete_all,
    }

    if chosen_backend == "sqlite":
        return _delete_from_sqlite(
            db_file=_resolve_sqlite_path(sqlite_path),
            effective_delete_all=effective_delete_all,
            clean_source=clean_source,
            source_candidates=source_candidates,
            clean_category=clean_category,
            clean_keyword=clean_keyword,
            clean_days=clean_days,
            clean_limit=clean_limit,
            order=order,
            applied_filters=applied_filters,
        )

    return _delete_from_postgres(
        effective_delete_all=effective_delete_all,
        clean_source=clean_source,
        source_candidates=source_candidates,
        clean_category=clean_category,
        clean_keyword=clean_keyword,
        clean_days=clean_days,
        clean_limit=clean_limit,
        order=order,
        applied_filters=applied_filters,
    )       
            
            
