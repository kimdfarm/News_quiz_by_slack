import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from psycopg2.extras import Json

from database import clean_old_news, get_db_connection

DEFAULT_SQLITE_PATH = os.getenv("NEWS_SQLITE_PATH", "news_store.sqlite")
STATE_FILE_PATH = Path(__file__).resolve().parent.parent / "storage_mode_state.json"


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
        results = []
        for r in rows:
            results.append({
                "title": r[0],
                "facts": r[1] or "",
                "opinion": r[2] or "",
                "stance_type": r[3] or "중립",
                "keywords": _parse_keywords_list(r[4]),
                "source": r[5] or "",
                "published_at": r[6].isoformat() if isinstance(r[6], datetime) else (str(r[6]) if r[6] else ""),
            })
        return results
    finally:
        cur.close()
        conn.close()


def _query_sqlite_news(
    category: str,
    limit: int,
    days: int = 7,
    db_path: str = DEFAULT_SQLITE_PATH,
) -> List[Dict[str, Any]]:
    if not os.path.exists(db_path):
        return []

    from crawler.sqlite_store import get_sqlite_conn

    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat()
    conn = get_sqlite_conn(db_path=db_path, read_only=True)
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
        results = []
        for r in rows:
            results.append({
                "title": r["title"],
                "facts": r["summary_facts"] or "",
                "opinion": r["author_opinion"] or "",
                "stance_type": r["stance_type"] or "중립",
                "keywords": _parse_keywords_list(r["keywords_json"]),
                "source": r["source_name"] or "",
                "published_at": r["published_at"] or "",
            })
        return results
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
    if active_backend == "sqlite":
        articles = _query_sqlite_news(category=category, limit=limit, days=days, db_path=sqlite_path)
    else:
        articles = _query_postgres_news(category=category, limit=limit, days=days)

    return {
        "storage_backend": active_backend,
        "sqlite_path": sqlite_path if active_backend == "sqlite" else None,
        "category": category,
        "count": len(articles),
        "articles": articles,
    }


def _resolve_sqlite_path(sqlite_path: Optional[str] = None) -> str:
    raw = (sqlite_path or os.getenv("NEWS_SQLITE_PATH") or DEFAULT_SQLITE_PATH).strip()
    return os.path.abspath(raw)


def _ensure_sqlite_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );
            """
        )
        cur.execute(
            """
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
            """
        )
        cur.execute(
            """
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
            issue_title, representative_summary, category_id, keywords_json,
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

        for r in rows:
            url = (r.get("url") or "").strip()
            if not url:
                skipped += 1
                continue

            pg_cur.execute("SELECT 1 FROM news_perspectives WHERE url = %s;", (url,))
            if pg_cur.fetchone():
                skipped += 1
                continue

            cat_id = _get_or_create_pg_category(pg_cur, r.get("category_name"))
            src_id = _get_or_create_pg_source(pg_cur, r.get("source_name"))
            issue_data = (
                {
                    "issue_title": r.get("issue_title"),
                    "representative_summary": r.get("representative_summary"),
                    "keywords": _parse_keywords_list(r.get("keywords_json")),
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
                    Json(meta_obj),
                    _parse_timestamp(r.get("published_at")),
                    _parse_timestamp(r.get("created_at")),
                ),
            )
            if pg_cur.rowcount and pg_cur.rowcount > 0:
                inserted += 1
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

        for r in rows:
            url = (r.get("url") or "").strip()
            if not url:
                skipped += 1
                continue

            sq_cur.execute("SELECT 1 FROM news_perspectives WHERE url = ?;", (url,))
            if sq_cur.fetchone():
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
                    "published_at": r.get("issue_published_at").isoformat() if isinstance(r.get("issue_published_at"), datetime) else r.get("issue_published_at"),
                    "created_at": r.get("issue_created_at").isoformat() if isinstance(r.get("issue_created_at"), datetime) else r.get("issue_created_at"),
                    "updated_at": r.get("issue_updated_at").isoformat() if isinstance(r.get("issue_updated_at"), datetime) else r.get("issue_updated_at"),
                }
                if r.get("issue_title") or r.get("issue_hash")
                else None
            )
            issue_id = _get_or_create_sqlite_issue(sq_cur, issue_data, cat_id)
            meta_json = json.dumps(_parse_json_obj(r.get("metadata")), ensure_ascii=False)
            pub_val = r.get("published_at")
            crt_val = r.get("created_at")

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
                    pub_val.isoformat() if isinstance(pub_val, datetime) else pub_val,
                    crt_val.isoformat() if isinstance(crt_val, datetime) else crt_val,
                ),
            )
            if sq_cur.rowcount and sq_cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

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