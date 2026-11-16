import json
from typing import List, Dict, Any
from sqlite_store import get_sqlite_conn, DEFAULT_SQLITE_PATH


def load_issues_with_perspectives(limit: int = 10, db_path: str = DEFAULT_SQLITE_PATH) -> List[Dict[str, Any]]:
    """DB 이관 없이 SQLite 파일을 직접 읽어 이슈별 팩트와 기자별 의견을 반환합니다."""
    conn = get_sqlite_conn(db_path=db_path, read_only=True)
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT i.id, i.issue_title, i.representative_summary, i.keywords_json, i.published_at,
                   c.name AS category_name
            FROM news_issues i
            LEFT JOIN categories c ON c.id = i.category_id
            ORDER BY i.id DESC
            LIMIT ?;
        """, (limit,))
        issues = []
        for row in cur.fetchall():
            issue_id = row["id"]
            cur.execute("""
                SELECT p.title, p.author_name, p.url, p.summary_facts, p.author_opinion, p.stance_type,
                       s.name AS source_name
                FROM news_perspectives p
                LEFT JOIN sources s ON s.id = p.source_id
                WHERE p.issue_id = ?
                ORDER BY p.id ASC;
            """, (issue_id,))
            perspectives = [dict(p) for p in cur.fetchall()]
            issues.append({
                "issue_id": issue_id,
                "title": row["issue_title"],
                "category": row["category_name"],
                "summary": row["representative_summary"],
                "keywords": json.loads(row["keywords_json"] or "[]"),
                "published_at": row["published_at"],
                "perspectives": perspectives,
            })
        return issues
    finally:
        cur.close()
        conn.close()