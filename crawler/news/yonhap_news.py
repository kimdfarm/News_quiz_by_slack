import sys
import os

# 상위 폴더(crawler/) 및 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rss_base_crawler import crawl_and_save_rss

def run():
    # 연합뉴스 주요 IT/과학 RSS 피드 주소
    rss_url = "https://www.yonhapnewstv.co.kr/category/news/it/feed"
    crawl_and_save_rss(rss_url, category="IT", source_name="연합뉴스TV")

if __name__ == "__main__":
    run()