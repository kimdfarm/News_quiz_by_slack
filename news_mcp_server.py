import json
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from database import get_db_connection, clean_old_news

# MCP 서버 인스턴스 생성
server = Server("news-quiz-mcp")

@server.list_tools()
async def list_tools() -> list:
    """
    LLM(모델)이 호출할 수 있는 툴(Tool) 목록을 정의하고 반환합니다.
    """
    return [
        {
            "name": "get_recent_news_for_quiz",
            "description": "최근 1주일 이내에 파이프라인 처리가 완료된 특정 카테고리의 뉴스 데이터를 조회합니다.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "조회할 뉴스 분야 (예: 경제, IT)"},
                    "limit": {"type": "integer", "description": "조회할 기사 개수 (기본값: 3)"}
                },
                "required": ["category"]
            }
        }
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    """
    LLM이 특정 툴을 호출했을 때 실제로 실행되는 로직입니다.
    """
    if name == "get_recent_news_for_quiz":
        # 1. 툴이 호출될 때마다 1주일 지난 오래된 데이터를 자동 정리 (TTL 적용)
        clean_old_news() 
        
        category = arguments.get("category", "경제")
        limit = arguments.get("limit", 3)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 2. 인덱스가 걸린 컬럼들을 활용하여 최신 완료된 뉴스 조회
        cur.execute("""
            SELECT title, summary_facts, author_opinion, keywords, source, published_at
            FROM news_articles 
            WHERE category = %s 
              AND processing_status = 'COMPLETED'
              AND published_at >= NOW() - INTERVAL '7 days'
            ORDER BY published_at DESC 
            LIMIT %s;
        """, (category, limit))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # 3. 조회된 결과를 JSON 포맷의 텍스트 형태로 변환 준비
        articles = [
            {
                "title": r[0],
                "facts": r[1],
                "opinion": r[2],
                "keywords": r[3],
                "source": r[4],
                "published_at": str(r[5])
            } 
            for r in rows
        ]
        
        return [{"type": "text", "text": json.dumps(articles, ensure_ascii=False)}]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Stdio 방식을 통해 MCP 서버를 실행하고 클라이언트 통신을 대기합니다."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())