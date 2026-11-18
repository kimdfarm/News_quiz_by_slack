import contextlib
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (
    _ROOT_DIR,
    _THIS_DIR,
    os.path.join(_ROOT_DIR, "crawler"),
    os.path.join(_ROOT_DIR, "crawler", "news"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from mcp.server.mcpserver import MCPServer

from crawler.news.plan_sources_from_request import plan_sources_from_request
from crawler.rss_base_crawler import crawl_multiple_rss_sources
from storage_backend import (
    DEFAULT_SQLITE_PATH,
    fetch_recent_news,
    get_active_storage_backend,
    set_active_storage_backend,
)

mcp = MCPServer("news-quiz-mcp")


@mcp.tool()
def set_storage_backend(storage_backend: str) -> str:
    """저장소 백엔드(postgres 또는 sqlite)를 변경합니다."""
    updated = set_active_storage_backend(storage_backend)
    return json.dumps(
        {
            "status": "ok",
            "STORAGE_BACKEND": updated,
            "sqlite_path": DEFAULT_SQLITE_PATH if updated == "sqlite" else None,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def get_recent_news_for_quiz(
    category: str = "경제",
    limit: int = 3,
    storage_backend: Optional[str] = None,
    sqlite_path: Optional[str] = None,
) -> str:
    """최근 뉴스 데이터를 조회합니다."""
    path = sqlite_path or os.getenv("NEWS_SQLITE_PATH", DEFAULT_SQLITE_PATH)
    result = fetch_recent_news(
        category=category,
        limit=limit,
        days=7,
        storage_backend=storage_backend,
        sqlite_path=path,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def run_rss_sources(
    sources_config: List[Dict[str, Any]],
    limit: Optional[int] = None,
    storage_mode: Optional[str] = None,
    sqlite_path: Optional[str] = None,
) -> str:
    """추론된 sources_config와 limit, storage_mode로 crawl_multiple_rss_sources를 실행합니다."""
    from crawler.rss_base_crawler import crawl_multiple_rss_sources

    path = sqlite_path or os.getenv("NEWS_SQLITE_PATH", DEFAULT_SQLITE_PATH)
    total = crawl_multiple_rss_sources(
        sources_config=sources_config,
        limit=limit,
        storage_mode=storage_mode,
        sqlite_path=path,
    )
    return json.dumps({"processed_total": total}, ensure_ascii=False)


import contextlib
import sys
import traceback

@mcp.tool()
def run_rss_by_request(
    request_text: str,
    storage_mode: Optional[str] = None,
    sqlite_path: Optional[str] = None,
    
) -> str:
    """자연어 요청을 해석해 요청 맞춤형 sources_config와 limit를 구성하고 crawl_multiple_rss_sources를 실행합니다."""
    try:

        sq_path = sqlite_path or os.getenv("NEWS_SQLITE_PATH", DEFAULT_SQLITE_PATH)

        with contextlib.redirect_stdout(sys.stderr):
            planned = plan_sources_from_request(
                request_text=request_text,
                storage_mode=storage_mode,
                sqlite_path=sq_path,
            )
            configs = planned.get("configs") or []
            effective_mode = planned.get("storage_mode") or storage_mode

            total_processed = crawl_multiple_rss_sources(
                sources_config=configs,
                limit=planned.get("default_limit"),
                storage_mode=effective_mode,
                sqlite_path=sq_path,
            )

        payload = {
            "request_text": request_text,
            "storage_mode": effective_mode,
            "planned_sources": [
                {
                    "source_name": c["source_name"],
                    "category": c["category"],
                    "limit": c.get("limit"),
                    "rss_urls": c["rss_urls"],
                }
                for c in configs
            ],
            "total_processed": total_processed,
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"run_rss_by_request 실패 ({type(e).__name__}): {e}") from e

@mcp.tool()
def export_sqlite_file(sqlite_path: Optional[str] = None) -> str:
    """SQLite DB 파일(news_store.sqlite)의 경로와 파일 정보를 확인해 다운로드용 메타데이터를 반환합니다."""
    default_file = os.path.join(_ROOT_DIR, "SQLITE", "news_store.sqlite")
    raw_path = (sqlite_path or os.getenv("NEWS_SQLITE_PATH") or default_file).strip()

    # 상대경로나 파일명만 들어온 경우 SQLITE 폴더 기준으로 보정
    if not os.path.isabs(raw_path) and not os.path.exists(raw_path):
        candidate = os.path.join(_ROOT_DIR, "SQLITE", os.path.basename(raw_path))
        if os.path.exists(candidate):
            raw_path = candidate

    abs_path = os.path.abspath(raw_path)
    if not os.path.isfile(abs_path):
        return json.dumps(
            {
                "status": "error",
                "message": f"SQLite 파일을 찾을 수 없습니다: {abs_path}",
                "file_path": abs_path,
            },
            ensure_ascii=False,
        )

    stat = os.stat(abs_path)
    return json.dumps(
        {
            "status": "ok",
            "file_path": abs_path,
            "filename": os.path.basename(abs_path),
            "size_bytes": stat.st_size,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def get_storage_status(sqlite_path: Optional[str] = None) -> str:
    """현재 활성 저장소 모드와 SQLite / Postgres 연결 및 파일 상태를 반환합니다."""
    from datetime import datetime
    from storage_backend import (
        DEFAULT_SQLITE_PATH as BACKEND_DEFAULT_SQLITE,
        get_active_storage_backend,
    )

    active_mode = get_active_storage_backend()
    target_sqlite = os.path.abspath((sqlite_path or os.getenv("NEWS_SQLITE_PATH") or BACKEND_DEFAULT_SQLITE).strip())

    sqlite_info: Dict[str, Any] = {
        "path": target_sqlite,
        "exists": os.path.isfile(target_sqlite),
        "size_bytes": 0,
        "updated_at": None,
    }
    if sqlite_info["exists"]:
        st = os.stat(target_sqlite)
        sqlite_info["size_bytes"] = st.st_size
        sqlite_info["updated_at"] = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    db_url = (os.getenv("DATABASE_URL") or "").strip()
    postgres_info: Dict[str, Any] = {
        "configured": bool(db_url),
        "connected": False,
        "error": None,
    }
    if db_url:
        try:
            from database import get_db_connection

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1;")
            cur.fetchone()
            cur.close()
            conn.close()
            postgres_info["connected"] = True
        except Exception as e:
            postgres_info["error"] = str(e)

    return json.dumps(
        {
            "status": "ok",
            "active_backend": active_mode,
            "sqlite": sqlite_info,
            "postgres": postgres_info,
        },
        ensure_ascii=False,
    )

@mcp.tool()
def get_storage_stats(
    target_backend: Optional[str] = None,
    sqlite_path: Optional[str] = None,
) -> str:
    """현재 또는 지정된 저장소(sqlite / postgres)의 기사 통계(총 건수, 언론사별/분야별 건수, 최근 수집 시각)를 반환합니다."""
    import sqlite3
    from storage_backend import (
        DEFAULT_SQLITE_PATH as BACKEND_DEFAULT_SQLITE,
        get_active_storage_backend,
        normalize_storage_backend,
    )

    active_mode = get_active_storage_backend()
    if target_backend and str(target_backend).strip():
        chosen_mode = normalize_storage_backend(target_backend)
    else:
        chosen_mode = active_mode

    if chosen_mode == "sqlite":
        db_file = os.path.abspath(
            (sqlite_path or os.getenv("NEWS_SQLITE_PATH") or BACKEND_DEFAULT_SQLITE).strip()
        )
        if not os.path.isfile(db_file):
            return json.dumps(
                {
                    "status": "unavailable",
                    "backend": "sqlite",
                    "active_backend": active_mode,
                    "message": "SQLite 저장소가 아직 준비되지 않았어요.",
                },
                ensure_ascii=False,
            )

        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM news_perspectives;")
            total_count = int(cur.fetchone()[0] or 0)

            cur.execute(
                """
                SELECT COALESCE(s.name, '미상') AS src, COUNT(*) AS cnt
                FROM news_perspectives p
                LEFT JOIN sources s ON p.source_id = s.id
                GROUP BY src
                ORDER BY cnt DESC
                LIMIT 5;
                """
            )
            by_source = [{"name": r[0], "count": int(r[1])} for r in cur.fetchall()]

            cur.execute(
                """
                SELECT COALESCE(c.name, '미분류') AS cat, COUNT(*) AS cnt
                FROM news_perspectives p
                LEFT JOIN news_issues i ON p.issue_id = i.id
                LEFT JOIN categories c ON i.category_id = c.id
                GROUP BY cat
                ORDER BY cnt DESC
                LIMIT 5;
                """
            )
            by_category = [{"name": r[0], "count": int(r[1])} for r in cur.fetchall()]

            cur.execute(
            """
            SELECT
                MIN(COALESCE(published_at, created_at)),
                MAX(COALESCE(published_at, created_at))
            FROM news_perspectives;
            """
        )
            oldest_at, latest_at = cur.fetchone()
        finally:
            cur.close()
            conn.close()

        return json.dumps(
            {
                "status": "ok",
                "backend": "sqlite",
                "active_backend": active_mode,
                "total_count": total_count,
                "by_source": by_source,
                "by_category": by_category,
                "latest_published_at": str(latest_at) if latest_at else None,
                "oldest_published_at": str(oldest_at) if oldest_at else None,
            },
            ensure_ascii=False,
        )

    # postgres
    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if not db_url:
        return json.dumps(
            {
                "status": "unavailable",
                "backend": "postgres",
                "active_backend": active_mode,
                "message": "Neon DB(PostgreSQL) 연결 정보가 설정되지 않았어요.",
            },
            ensure_ascii=False,
        )

    try:
        from database import get_db_connection

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM news_perspectives;")
            total_count = int(cur.fetchone()[0] or 0)

            cur.execute(
                """
                SELECT COALESCE(s.name, '미상') AS src, COUNT(*) AS cnt
                FROM news_perspectives p
                LEFT JOIN sources s ON p.source_id = s.id
                GROUP BY src
                ORDER BY cnt DESC
                LIMIT 5;
                """
            )
            by_source = [{"name": r[0], "count": int(r[1])} for r in cur.fetchall()]

            cur.execute(
                """
                SELECT COALESCE(c.name, '미분류') AS cat, COUNT(*) AS cnt
                FROM news_perspectives p
                LEFT JOIN news_issues i ON p.issue_id = i.id
                LEFT JOIN categories c ON i.category_id = c.id
                GROUP BY cat
                ORDER BY cnt DESC
                LIMIT 5;
                """
            )
            by_category = [{"name": r[0], "count": int(r[1])} for r in cur.fetchall()]

            cur.execute(
            """
            SELECT
                MIN(COALESCE(published_at::text, created_at::text)),
                MAX(COALESCE(published_at::text, created_at::text))
            FROM news_perspectives;
            """
        )
            oldest_at, latest_at = cur.fetchone()
        finally:
            cur.close()
            conn.close()

        return json.dumps(
            {
                "status": "ok",
                "backend": "postgres",
                "active_backend": active_mode,
                "total_count": total_count,
                "by_source": by_source,
                "by_category": by_category,
                "oldest_published_at": str(oldest_at) if oldest_at else None,
                "latest_published_at": str(latest_at) if latest_at else None,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "backend": "postgres",
                "active_backend": active_mode,
                "message": f"Neon DB 조회 실패: {e}",
            },
            ensure_ascii=False,
        )

@mcp.tool()
def sync_sqlite_to_postgres(
    limit: Optional[int] = None,
    sqlite_path: Optional[str] = None,
) -> str:
    """로컬 SQLite(news_store.sqlite)의 기사를 Neon DB(PostgreSQL)로 동기화합니다.
    limit을 지정하면 최근 N건만, 지정하지 않으면 전체를 대상으로 하며 중복 URL은 건너뜁니다.
    """
    from storage_backend import sync_sqlite_to_postgres as _sync_sq_to_pg

    result = _sync_sq_to_pg(limit=limit, sqlite_path=sqlite_path)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def sync_postgres_to_sqlite(
    limit: Optional[int] = None,
    sqlite_path: Optional[str] = None,
) -> str:
    """Neon DB(PostgreSQL)의 기사를 로컬 SQLite(news_store.sqlite)로 동기화합니다.
    limit을 지정하면 최근 N건만, 지정하지 않으면 전체를 대상으로 하며 중복 URL은 건너뜁니다.
    """
    from storage_backend import sync_postgres_to_sqlite as _sync_pg_to_sq

    result = _sync_pg_to_sq(limit=limit, sqlite_path=sqlite_path)
    return json.dumps(result, ensure_ascii=False)









if __name__ == "__main__":
    mcp.run(transport="stdio")