import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rss_base_crawler import crawl_and_save_rss

def run():
    # AITIMES 인공지능/테크 전문 RSS 피드 주소
    rss_url = "https://www.aitimes.com/rss/allArticle.xml"
    crawl_and_save_rss(rss_url, category="AI/테크", source_name="AITIMES")

if __name__ == "__main__":
    run()