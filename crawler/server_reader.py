import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from database import get_db_connection
from sqlite_store import DEFAULT_SQLITE_PATH, get_sqlite_conn


def _normalize_keywords(raw: Any) -> List[str]:
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


def _normalize_metadata(raw: Any) -> Dict[str, Any]:
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


def _format_dt(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _load_issues_from_postgres(limit: int = 10) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT i.id, i.issue_title, i.representative_summary, i.keywords, i.published_at,
                   c.name AS category_name
            FROM news_issues i
            LEFT JOIN categories c ON c.id = i.category_id
            ORDER BY i.id DESC
            LIMIT %s;
            """,
            (limit,),
        )
        issue_rows = cur.fetchall()
        issues: List[Dict[str, Any]] = []

        for row in issue_rows:
            issue_id = row[0]
            cur.execute(
                """
                SELECT p.id, p.title, p.author_name, p.url, p.summary_facts,
                       p.author_opinion, p.stance_type, p.metadata, p.published_at,
                       s.name AS source_name
                FROM news_perspectives p
                LEFT JOIN sources s ON s.id = p.source_id
                WHERE p.issue_id = %s
                ORDER BY p.id ASC;
                """,
                (issue_id,),
            )
            perspectives = [
                {
                    "id": p[0],
                    "title": p[1],
                    "author_name": p[2],
                    "url": p[3],
                    "summary_facts": p[4],
                    "author_opinion": p[5],
                    "stance_type": p[6],
                    "metadata": _normalize_metadata(p[7]),
                    "published_at": _format_dt(p[8]),
                    "source_name": p[9],
                }
                for p in cur.fetchall()
            ]
            issues.append({
                "issue_id": issue_id,
                "title": row[1],
                "category": row[5],
                "summary": row[2],
                "keywords": _normalize_keywords(row[3]),
                "published_at": _format_dt(row[4]),
                "perspectives": perspectives,
            })

        return issues
    finally:
        cur.close()
        conn.close()


def _load_issues_from_sqlite(limit: int = 10, db_path: str = DEFAULT_SQLITE_PATH) -> List[Dict[str, Any]]:
    conn = get_sqlite_conn(db_path=db_path, read_only=True)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT i.id, i.issue_title, i.representative_summary, i.keywords_json, i.published_at,
                   c.name AS category_name
            FROM news_issues i
            LEFT JOIN categories c ON c.id = i.category_id
            ORDER BY i.id DESC
            LIMIT ?;
            """,
            (limit,),
        )
        issues: List[Dict[str, Any]] = []
        for row in cur.fetchall():
            issue_id = row["id"]
            cur.execute(
                """
                SELECT p.id, p.title, p.author_name, p.url, p.summary_facts,
                       p.author_opinion, p.stance_type, p.metadata_json, p.published_at,
                       s.name AS source_name
                FROM news_perspectives p
                LEFT JOIN sources s ON s.id = p.source_id
                WHERE p.issue_id = ?
                ORDER BY p.id ASC;
                """,
                (issue_id,),
            )
            perspectives = [
                {
                    "id": p["id"],
                    "title": p["title"],
                    "author_name": p["author_name"],
                    "url": p["url"],
                    "summary_facts": p["summary_facts"],
                    "author_opinion": p["author_opinion"],
                    "stance_type": p["stance_type"],
                    "metadata": _normalize_metadata(p["metadata_json"]),
                    "published_at": _format_dt(p["published_at"]),
                    "source_name": p["source_name"],
                }
                for p in cur.fetchall()
            ]
            issues.append({
                "issue_id": issue_id,
                "title": row["issue_title"],
                "category": row["category_name"],
                "summary": row["representative_summary"],
                "keywords": _normalize_keywords(row["keywords_json"]),
                "published_at": _format_dt(row["published_at"]),
                "perspectives": perspectives,
            })
        return issues
    finally:
        cur.close()
        conn.close()


def load_issues_with_perspectives(
    limit: int = 10,
    db_path: str = DEFAULT_SQLITE_PATH,
    storage_backend: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    database.py 스키마 기준으로 이슈별 팩트와 기자별 관점을 조회합니다.
    storage_backend가 'sqlite'면 SQLite 파일을 읽고, 그 외('postgres' 또는 None)는 Neon DB를 읽습니다.
    """
    mode = (storage_backend or "").strip().lower()
    if mode == "sqlite":
        return _load_issues_from_sqlite(limit=limit, db_path=db_path)
    return _load_issues_from_postgres(limit=limit)