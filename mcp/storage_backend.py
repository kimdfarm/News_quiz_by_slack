import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _query_postgres_news(category: str, limit: int, days: int = 7) -> List[Dict[str, Any]]:
    clean_old_news()
    cutoff = datetime.now() - timedelta(days=days)

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT p.title, p.summary_facts, p.author_opinion, p.stance_type,
                   i.keywords_json, s.name AS source_name, p.published_at
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
            kw_raw = r[4]
            if isinstance(kw_raw, list):
                keywords = kw_raw
            elif isinstance(kw_raw, str) and kw_raw.strip():
                try:
                    keywords = json.loads(kw_raw)
                except Exception:
                    keywords = [kw_raw]
            else:
                keywords = []

            results.append({
                "title": r[0],
                "facts": r[1] or "",
                "opinion": r[2] or "",
                "stance_type": r[3] or "중립",
                "keywords": keywords,
                "source": r[5] or "",
                "published_at": str(r[6]) if r[6] else "",
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
                   i.keywords_json, s.name AS source_name, p.published_at
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
            try:
                keywords = json.loads(r["keywords_json"] or "[]")
            except Exception:
                keywords = []
            results.append({
                "title": r["title"],
                "facts": r["summary_facts"] or "",
                "opinion": r["author_opinion"] or "",
                "stance_type": r["stance_type"] or "중립",
                "keywords": keywords,
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