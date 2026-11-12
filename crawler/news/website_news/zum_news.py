import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rss_base_crawler import crawl_and_save_rss

def run():
    # 줌(ZUM) 뉴스 주요 섹션 또는 구글 검색 기반 줌 타겟 RSS 피드 주소
    # (포털 애그리게이터 성격을 활용하기 위한 쿼리 혹은 줌 뉴스 피드 경로)
    rss_url = "https://news.google.com/rss/search?q=site:zum.com+경제+OR+IT&hl=ko&gl=KR&ceid=KR:ko"
    crawl_and_save_rss(rss_url, category="종합/트렌드", source_name="ZUM뉴스(Aggregator)")

if __name__ == "__main__":
    run()