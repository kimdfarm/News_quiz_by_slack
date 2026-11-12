import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rss_base_crawler import crawl_and_save_rss

def run():
    # 조선일보 주요 뉴스 RSS 피드 주소 (예: 경제 또는 종합 섹션)
    rss_url = "https://www.chosun.com/arc/outboundlib/rss/?outputType=xml"
    crawl_and_save_rss(rss_url, category="종합/경제", source_name="조선일보")

if __name__ == "__main__":
    run()