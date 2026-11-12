import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rss_base_crawler import crawl_and_save_rss

def run():
    # MBC 뉴스 종합 RSS 피드 주소
    rss_url = "https://imnews.imbc.com/rss/news/news_00.xml"
    crawl_and_save_rss(rss_url, category="종합/사회", source_name="MBC뉴스")

if __name__ == "__main__":
    run()