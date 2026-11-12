import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rss_base_crawler import crawl_and_save_rss

def run():
    # 블로터 IT/스타트업 RSS 피드 주소
    rss_url = "https://www.bloter.net/rss/all.xml"
    crawl_and_save_rss(rss_url, category="IT/트렌드", source_name="블로터")

if __name__ == "__main__":
    run()