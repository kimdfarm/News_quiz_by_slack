from news.yonhap_news import run as run_yonhap
from news.hani_news import run as run_hani
from news.mk_news import run as run_mk
from news.chosun_news import run as run_chosun
from news.donga_news import run as run_donga

if __name__ == "__main__":
    print("=== 🚀 전체 언론사 뉴스 수집 배치 시작 ===")
    run_yonhap()
    run_hani()
    run_mk()
    run_chosun()
    run_donga()
    print("=== ✨ 모든 뉴스 수집 및 중복 필터링 적재 완료 ===")