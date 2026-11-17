from contextlib import redirect_stdout
import os
import json
import sys
from typing import Dict, List, Optional, Union, Any

# 기본 매핑: 언론사명 -> {"urls": [...], "category": "..."}
DEFAULT_PRESS_RSS_MAP: Dict[str, Dict[str, Any]] = {
    # --- 연합 (연합뉴스 / 연합뉴스TV 분야별) ---
    "연합/종합": {
        "urls": ["https://www.yonhapnewstv.co.kr/browse/feed/"],
        "category": "종합",
    },
    "연합/경제": {
        "urls": ["https://www.yna.co.kr/rss/economy.xml"],
        "category": "경제",
    },
    "연합/산업_IT": {
        "urls": ["https://www.yna.co.kr/rss/industry.xml"],
        "category": "IT/산업",
    },
    "연합/정치": {
        "urls": ["https://www.yna.co.kr/rss/politics.xml"],
        "category": "정치",
    },
    "연합/사회": {
        "urls": ["https://www.yna.co.kr/rss/society.xml"],
        "category": "사회",
    },
    "연합/세계": {
        "urls": ["https://www.yna.co.kr/rss/international.xml"],
        "category": "세계",
    },

    # --- 경향신문 (분야별) ---
    "경향/종합": {
        "urls": ["https://www.khan.co.kr/rss/rssdata/total_news.xml"],
        "category": "종합",
    },
    "경향/경제": {
        "urls": ["https://www.khan.co.kr/rss/rssdata/economy_news.xml"],
        "category": "경제",
    },
    "경향/정치": {
        "urls": ["https://www.khan.co.kr/rss/rssdata/politic_news.xml"],
        "category": "정치",
    },
    "경향/사회": {
        "urls": ["https://www.khan.co.kr/rss/rssdata/society_news.xml"],
        "category": "사회",
    },
    "경향/IT_과학": {
        "urls": ["https://www.khan.co.kr/rss/rssdata/science_news.xml"],
        "category": "IT/과학",
    },

    # --- 매일경제 (경제·산업·IT 가치 높음) ---
    "매경/경제": {
        "urls": ["https://www.mk.co.kr/rss/30100041/"],
        "category": "경제",
    },
    "매경/기업_경영": {
        "urls": ["https://www.mk.co.kr/rss/50100032/"],
        "category": "기업/산업",
    },

    # --- AI · IT · 테크 전문 (관점·분석·기술 트렌드 가치 높음) ---
    "AITIMES": {
        "urls": ["https://www.aitimes.com/rss/allArticle.xml"],
        "category": "AI/테크",
    },
    "전자신문/전체": {
        "urls": ["https://rss.etnews.com/Section901.xml"],
        "category": "IT/산업",
    },
    "테크M": {
        "urls": ["https://www.techm.kr/rss/allArticle.xml"],
        "category": "IT/테크",
    },

    # --- 주요 일간지 · 방송 (여론·논조 비교용) ---
    "한겨레/전체": {
        "urls": ["https://www.hani.co.kr/rss/"],
        "category": "종합",
    },
    "조선일보/전체": {
        "urls": ["https://www.chosun.com/arc/outboundfeeds/rss/?outputType=xml"],
        "category": "종합",
    },
    
    "SBS/뉴스": {
        "urls": ["https://news.sbs.co.kr/news/headlineRssFeed.do?plink=RSSREADER"],
        "category": "종합",
    },

    # --- 조선일보 / 조선비즈 / 조세일보 / 한국일보 ---
    "조선/정치": {
        "urls": ["https://www.chosun.com/arc/outboundfeeds/rss/category/politics/?outputType=xml"],
        "category": "정치",
    },
    "조선비즈/전체": {
        "urls": ["https://biz.chosun.com/arc/outboundfeeds/rss/?outputType=xml"],
        "category": "경제/산업",
    },
    "조세일보/전체": {
        "urls": ["https://www.joseilbo.com/Contents/rss/rss_total.php"],
        "category": "경제/세무",
    },
    "뉴시스/속보": {
        "urls": ["https://www.newsis.com/RSS/sokbo.xml"],
        "category": "속보",
    },

    # --- 주요 일간지 추가 섹션 ---
    "조선/국제": {
        "urls": ["https://www.chosun.com/arc/outboundfeeds/rss/category/international/?outputType=xml"],
        "category": "국제",
    },
    # --- 동아일보  ---
    "동아/전체": {
        "urls": ["https://rss.donga.com/total.xml"],
        "category": "종합",
    },
    
    "동아/정치": {
            "urls": ["https://rss.donga.com/politics.xml"],
            "category": "정치",
        },
    "동아/사회": {
                "urls": ["https://rss.donga.com/national.xml"],
                "category": "사회",
            },
    "동아/경제": {
            "urls": ["https://rss.donga.com/economy.xml"],
            "category": "경제",
        },
    "동아/국제": {
                "urls": ["https://rss.donga.com/international.xml"],
                "category": "국제",
            },
    "동아/사설칼럼": {
                    "urls": ["https://rss.donga.com/editorials.xml"],
                    "category": "사설/칼럼",
                },
    "동아/의학과학": {
                        "urls": ["https://rss.donga.com/science.xml"],
                        "category": "의학과학",
                    },
    "동아/문화연예": {
                            "urls": ["https://rss.donga.com/culture.xml"],
                            "category": "문화연예",
                        },
    "동아/스포츠": {
                            "urls": ["https://rss.donga.com/sports.xml"],
                            "category": "스포츠",
                        },
    "동아/사람속으로": {
                            "urls": ["https://rss.donga.com/inmul.xml"],
                            "category": "사람속으로",
                        },
    "동아/건강": {
                            "urls": ["https://rss.donga.com/health.xml"],
                            "category": "건강",
                        },
    "동아/도서": {
                            "urls": ["https://rss.donga.com/book.xml"],
                            "category": "도서",
                        },
    "동아/공연": {
                            "urls": ["https://rss.donga.com/show.xml"],
                            "category": "공연",
                        },
    "동아/여성": {
                            "urls": ["https://rss.donga.com/woman.xml"],
                            "category": "여성",
                        },
    "동아/여행": {
                            "urls": ["https://rss.donga.com/travel.xml"],
                            "category": "여행",
                        },
    "동아/생활정보": {
                            "urls": ["https://rss.donga.com/lifeinfo.xml"],
                            "category": "생활정보",
                        },
    "동아/스포츠": {
                            "urls": ["https://rss.donga.com/sportsdonga/sports.xml"],
                            "category": "스포츠",
                        },
    "동아/야구/MLB": {
                            "urls": ["https://rss.donga.com/sportsdonga/baseball.xml"],
                            "category": "야구/MLB",
                        },
    "동아/축구": {
                            "urls": ["https://rss.donga.com/sportsdonga/soccer.xml"],
                            "category": "축구",
                        },
    "동아/골프": {
                            "urls": ["https://rss.donga.com/sportsdonga/golf.xml"],
                            "category": "골프",
                        },
    "동아/스포츠_일반": {
                            "urls": ["https://rss.donga.com/sportsdonga/sports_general.xml"],
                            "category": "스포츠_일반",
                        },
    "동아/e스포츠/게임": {
                            "urls": ["https://rss.donga.com/sportsdonga/sports_game.xml"],
                            "category": "e스포츠/게임",
                        },
    "동아/엔터테인먼트": {
                            "urls": ["https://rss.donga.com/sportsdonga/entertainment.xml"],
                            "category": "엔터테인먼트",
                        },

        
    
    
}


def load_press_rss_map() -> Dict[str, Dict[str, Any]]:
    """
    기본 매핑을 불러오고, 환경변수 RSS_SOURCES_JSON이 있으면 덮어쓰거나 추가합니다.
    예시:
    RSS_SOURCES_JSON='{"연합":["https://.../rss1.xml","https://.../rss2.xml"],"한국경제":{"urls":["https://.../rss.xml"],"category":"경제"}}'
    """
    merged: Dict[str, Dict[str, Any]] = {
        k: {"urls": list(v.get("urls") or []), "category": v.get("category", "종합")}
        for k, v in DEFAULT_PRESS_RSS_MAP.items()
    }

    raw_json = (os.getenv("RSS_SOURCES_JSON") or "").strip()
    if not raw_json:
        return merged

    try:
        override = json.loads(raw_json)
        if not isinstance(override, dict):
            return merged

        for press_name, spec in override.items():
            name = str(press_name).strip()
            if not name:
                continue

            if isinstance(spec, str):
                urls = [spec.strip()] if spec.strip() else []
                category = merged.get(name, {}).get("category", "종합")
            elif isinstance(spec, list):
                urls = [str(u).strip() for u in spec if str(u).strip()]
                category = merged.get(name, {}).get("category", "종합")
            elif isinstance(spec, dict):
                raw_urls = spec.get("urls") or spec.get("rss_urls") or []
                if isinstance(raw_urls, str):
                    urls = [raw_urls.strip()] if raw_urls.strip() else []
                else:
                    urls = [str(u).strip() for u in raw_urls if str(u).strip()]
                category = str(spec.get("category") or merged.get(name, {}).get("category") or "종합").strip()
            else:
                continue

            if urls:
                merged[name] = {"urls": urls, "category": category}
    except Exception as e:
        print(f"[!] 매핑되지 않은 언론사 스킵: {name}", file=sys.stderr)
        redirect_stdout(sys.stderr)

    return merged


def build_sources_config(
    press_names: Optional[List[str]] = None,
    limit: Optional[int] = None,
    category_override: Optional[str] = None,
    storage_mode: Optional[str] = None,
    sqlite_path: Optional[str] = None,
    custom_map: Optional[Dict[str, Union[str, List[str], Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    언론사 이름 목록과 옵션(limit, category_override, storage_mode 등)을 받아
    crawl_multiple_rss_sources에 바로 넣을 수 있는 sources_config 리스트로 변환합니다.
    """
    press_map = load_press_rss_map()

    # 호출 시점에 임시로 URL/카테고리를 변형하고 싶을 때 반영
    if custom_map:
        for k, v in custom_map.items():
            name = str(k).strip()
            if not name:
                continue
            if isinstance(v, str):
                press_map[name] = {"urls": [v.strip()], "category": press_map.get(name, {}).get("category", "종합")}
            elif isinstance(v, list):
                press_map[name] = {"urls": [u.strip() for u in v if u.strip()], "category": press_map.get(name, {}).get("category", "종합")}
            elif isinstance(v, dict):
                urls = v.get("urls") or v.get("rss_urls") or []
                if isinstance(urls, str):
                    urls = [urls.strip()]
                press_map[name] = {
                    "urls": [u.strip() for u in urls if u.strip()],
                    "category": v.get("category") or press_map.get(name, {}).get("category", "종합"),
                }

    targets = press_names if press_names else list(press_map.keys())
    configs: List[Dict[str, Any]] = []

    for raw_name in targets:
        name = str(raw_name).strip()
        entry = press_map.get(name)
        if not entry or not entry.get("urls"):
            print(f"[!] 매핑되지 않았거나 RSS URL이 없는 언론사 스킵: {name}")
            redirect_stdout(sys.stderr)
            continue

        cfg: Dict[str, Any] = {
            "source_name": name,
            "category": category_override or entry.get("category", "종합"),
            "rss_urls": entry["urls"],
        }
        if limit is not None:
            cfg["limit"] = limit
        if storage_mode is not None:
            cfg["storage_mode"] = storage_mode
        if sqlite_path is not None:
            cfg["sqlite_path"] = sqlite_path

        configs.append(cfg)

    return configs

if __name__ == "__main__":
    from crawler.rss_base_crawler import crawl_multiple_rss_sources

    # 1) 등록된 언론사 중 "연합", "경향"만 골라 각 30개씩 sqlite로 저장
    configs = build_sources_config(
        press_names=["연합", "경향"],
        limit=30,
        storage_mode="sqlite",
    )
    crawl_multiple_rss_sources(configs, limit=30, storage_mode="sqlite")

    # 2) 코드 수정 없이 이번 실행만 RSS URL이나 카테고리를 변형해서 실행
    modified_configs = build_sources_config(
        press_names=["연합"],
        limit=10,
        custom_map={
            "연합": {
                "urls": ["https://www.yonhapnewstv.co.kr/browse/feed/"],
                "category": "속보",
            }
        },
    )
    crawl_multiple_rss_sources(modified_configs, storage_mode="postgres")