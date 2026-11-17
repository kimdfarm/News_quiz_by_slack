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

if __name__ == "__main__":
    mcp.run(transport="stdio")