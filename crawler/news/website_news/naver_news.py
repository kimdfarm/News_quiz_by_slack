import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rss_base_crawler import crawl_and_save_rss

def run():
    # 네이버/오픈 뉴스 키워드(IT/경제) 통합 수집용 RSS 쿼리
    rss_url = "https://news.google.com/rss/search?q=IT+OR+경제+한국&hl=ko&gl=KR&ceid=KR:ko"
    crawl_and_save_rss(rss_url, category="종합/경제", source_name="네이버뉴스(Aggregator)")

if __name__ == "__main__":
    run()