import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rss_base_crawler import crawl_and_save_rss

def run():
    # 구글 뉴스 RSS를 활용해 '인공지능 OR 경제' 주요 최신 기사 통합 수집
    rss_url = "https://news.google.com/rss/search?q=인공지능+OR+경제+한국&hl=ko&gl=KR&ceid=KR:ko"
    crawl_and_save_rss(rss_url, category="AI/경제통합", source_name="Google뉴스(Aggregator)")

if __name__ == "__main__":
    run()