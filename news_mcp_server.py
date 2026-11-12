import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from database import get_db_connection, clean_old_news

server = Server("news-quiz-mcp")

@server.list_tools()
async def list_tools() -> list:
    return [
        {
            "name": "get_recent_news_for_quiz",
            "description": "최근 1주일 이내의 뉴스 데이터 중 특정 카테고리의 데이터를 조회합니다.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "조회할 뉴스 분야 (예: 경제, IT)"}
                },
                "required": ["category"]
            }
        }
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    if name == "get_recent_news_for_quiz":
        clean_old_news()  # 조회 시점에 1주일 경과 데이터 정리
        category = arguments.get("category", "경제")
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT title, summary_facts, author_opinion 
            FROM news_articles 
            WHERE category = %s AND created_at >= NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC LIMIT 3;
        """, (category,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        articles = [{"title": r[0], "facts": r[1], "opinion": r[2]} for r in rows]
        return [{"type": "text", "text": json.dumps(articles, ensure_ascii=False)}]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())