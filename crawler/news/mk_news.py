import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rss_base_crawler import crawl_and_save_rss

def run():
    # 매일경제 경제 뉴스 RSS 피드 주소 예시
    rss_url = "https://www.mk.co.kr/rss/30100041/" # 경제 일반 RSS
    crawl_and_save_rss(rss_url, category="경제", source_name="매일경제")

if __name__ == "__main__":
    run()