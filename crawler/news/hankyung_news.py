import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rss_base_crawler import crawl_and_save_rss

def run():
    # 한국경제 뉴스 전체/경제 RSS 피드 주소
    rss_url = "https://www.hankyung.com/feed/all"
    crawl_and_save_rss(rss_url, category="경제/산업", source_name="한국경제")

if __name__ == "__main__":
    run()