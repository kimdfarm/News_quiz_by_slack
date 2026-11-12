import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rss_base_crawler import crawl_and_save_rss

def run():
    # 다음 뉴스 및 주요 포털 실시간 이슈를 연동하기 위한 RSS 피드 엔드포인트
    rss_url = "https://news.google.com/rss/search?q=site:daum.net+경제+OR+IT&hl=ko&gl=KR&ceid=KR:ko"
    crawl_and_save_rss(rss_url, category="경제/IT트렌드", source_name="Daum뉴스(Aggregator)")

if __name__ == "__main__":
    run()