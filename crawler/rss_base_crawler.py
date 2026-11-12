import sys
import os
import requests
import feedparser
from datetime import datetime

# 상위 디렉토리(database.py) 임포트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import save_news_article

def crawl_and_save_rss(rss_url: str, category: str, source_name: str):
    """
    특정 언론사나 플랫폼의 RSS를 수집하여 DB에 저장합니다.
    (database.py의 url UNIQUE 및 content_hash 덕분에 중복 기사는 자동 필터링됩니다.)
    """
    print(f"[{source_name}] 크롤링 파이프라인 구동 중... ({rss_url})")
    
    try:
        feed = feedparser.parse(rss_url)
        count = 0
        
        for entry in feed.entries:
            title = entry.get("title", "제목 없음")
            url = entry.get("link", "")
            summary = entry.get("summary", title)
            
            # 발행일 파싱
            published_at = datetime.now()
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6])
                
            article_data = {
                "title": title,
                "summary_facts": summary,
                "author_opinion": "", # 상세 페이지 파싱 시 추가 가능
                "category": category,
                "keywords": [category, source_name],
                "source": source_name,
                "url": url,
                "author_name": entry.get("author", "기자 미상"),
                "published_at": published_at,
                "raw_content": summary,
                "model_version": "gemma:3.4b",
                "metadata": {"collector": "rss_parser"}
            }
            
            # DB 저장 시도 (중복 URL은 DB 레벨에서 차단됨)
            save_news_article(article_data)
            count += 1
            
        print(f"[{source_name}] 총 {count}개 기사 처리 완료 (중복 제외 적재)")
        
    except Exception as e:
        print(f"[{source_name}] 크롤링 중 오류 발생: {e}")