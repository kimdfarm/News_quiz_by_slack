import sys
import os
import re
import json
import html
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Union, List, Optional

import requests
import feedparser
from bs4 import BeautifulSoup
import trafilatura
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()
DEFAULT_STORAGE_BACKEND = (os.getenv("STORAGE_BACKEND") or "postgres").strip().lower()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # news/의 상위 폴더
SQLITE_DIR = os.path.join(PROJECT_ROOT, "SQLITE")

_raw_sqlite_name = os.path.basename((os.getenv("NEWS_SQLITE_PATH") or "news_store.sqlite").strip())
if not _raw_sqlite_name.endswith((".sqlite", ".db", ".sqlite3")):
    _raw_sqlite_name = f"{_raw_sqlite_name}.sqlite"

DEFAULT_SQLITE_PATH = os.path.join(SQLITE_DIR, _raw_sqlite_name)
os.makedirs(SQLITE_DIR, exist_ok=True)

try:
    from crawler.news.storage_mode_config import (
        get_persisted_storage_mode,
        normalize_storage_mode,
    )
except ImportError:
    from news.storage_mode_config import (
        get_persisted_storage_mode,
        normalize_storage_mode,
    )


def _resolve_storage_mode(storage_mode: Optional[str] = None) -> str:
    explicit = normalize_storage_mode(storage_mode)
    if explicit in ("sqlite", "postgres"):
        return explicit
    return get_persisted_storage_mode()

def _import_sqlite_store():
    candidate_dirs = [
        BASE_DIR,
        PROJECT_ROOT,
        os.path.join(BASE_DIR, "news"),
        os.path.join(PROJECT_ROOT, "crawler"),
        os.path.join(PROJECT_ROOT, "crawler", "news"),
    ]
    for d in candidate_dirs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    for mod_name in ("sqlite_store", "crawler.sqlite_store", "crawler.news.sqlite_store"):
        try:
            mod = __import__(mod_name, fromlist=["get_existing_urls_sqlite", "save_news_article_sqlite"])
            if hasattr(mod, "get_existing_urls_sqlite") and hasattr(mod, "save_news_article_sqlite"):
                return mod
        except ImportError:
            continue

    raise ModuleNotFoundError("sqlite_store 모듈을 찾을 수 없습니다. sqlite_store.py 위치를 확인해 주세요.")

def _resolve_sqlite_path(sqlite_path: Optional[str] = None) -> str:
    cleaned = (sqlite_path or "").strip()
    return cleaned if cleaned else DEFAULT_SQLITE_PATH

def _fetch_existing_urls_by_mode(urls: list, mode: str, sqlite_path: Optional[str]) -> set:
    try:
        if mode == "sqlite":
            effective_sqlite = _resolve_sqlite_path(sqlite_path)
            sqlite_store = _import_sqlite_store()
            result = sqlite_store.get_existing_urls_sqlite(urls, db_path=effective_sqlite)
        else:
            result = get_existing_urls(urls)
        return set(result) if result else set()
    except Exception as e:
        print(f"  [!] 기존 URL 조회 실패 ({mode}), 중복 체크를 건너뜁니다: {e}", file=sys.stderr)
        return set()


def _persist_article_by_mode(article_data: dict, mode: str, sqlite_path: Optional[str]):
    if mode == "sqlite":
        effective_sqlite = _resolve_sqlite_path(sqlite_path)
        sqlite_store = _import_sqlite_store()
        sqlite_store.save_news_article_sqlite(article_data, db_path=effective_sqlite)
        return
    save_news_article(article_data)

# 데이터 수집(크롤링) 실행 시에만 LangSmith 트레이싱 비활성화
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGSMITH_ENDPOINT"] = ""
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import save_news_article, compute_text_similarity, generate_content_hash,url_exists , get_db_connection
from langchain_core.runnables import RunnableLambda

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, text/html, */*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}
_CACHED_CHAIN = None
_CACHED_CHAIN_LABEL = None
MIN_LLM_CONTENT_CHARS = int(os.getenv("MIN_LLM_CONTENT_CHARS", "120"))

_LAST_USED_MODEL = None


def _wrap_with_model_tracker(chain, model_label: str):
    def _mark_success(output):
        global _LAST_USED_MODEL
        _LAST_USED_MODEL = model_label
        return output

    return chain | RunnableLambda(_mark_success)

def _get_or_build_chain():
    global _CACHED_CHAIN, _CACHED_CHAIN_LABEL
    if _CACHED_CHAIN is not None:
        return _CACHED_CHAIN, _CACHED_CHAIN_LABEL

    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "뉴스 기사 및 기자 관점 분석기입니다. 유효한 JSON만 출력하세요. "
            "키: summary_facts(문자열), author_opinion(문자열), "
            "stance_type('긍정'|'부정'|'비판'|'중립'), keywords(문자열 리스트, 최대 5개).",
        ),
        (
            "human",
            "제목: {title}\n"
            "본문: {content}\n\n"
            "1. summary_facts: 핵심 사실 요약(3문장 이내)\n"
            "2. author_opinion: 기자 관점·전망·비판('기자는 ~평가한다' 같은 설명투 지양, 없으면 \"\")\n"
            "3. stance_type: 긍정, 부정, 비판, 중립 중 1개\n"
            "4. keywords: 핵심 키워드(최대 5개)\n"
            "JSON 예시: {{\"summary_facts\":\"...\",\"author_opinion\":\"...\",\"stance_type\":\"중립\",\"keywords\":[\"...\"]}}",
        ),
    ])

    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "700"))
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    ollama_model_name = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    chains = []
    labels = []

    if groq_api_key:
        for model_name in _get_groq_model_queue():
            label = f"groq:{model_name}"
            groq_llm = ChatGroq(
                model=model_name,
                api_key=groq_api_key,
                temperature=0.1,
                max_tokens=groq_max_tokens,
            )
            groq_chain = (prompt | groq_llm | parser).with_config({"run_name": label})
            chains.append(_wrap_with_model_tracker(groq_chain, label))
            labels.append(label)
            
    if use_ollama:
        ollama_llm = ChatOllama(model=ollama_model_name, base_url=ollama_base_url, temperature=0.1)
        ollama_chain = (prompt | ollama_llm | parser).with_config({"run_name": f"ollama:{ollama_model_name}"})
        chains.append(ollama_chain)
        labels.append(f"ollama:{ollama_model_name}")

    if not chains:
        raise RuntimeError("GROQ_API_KEY가 설정되어 있지 않고 USE_OLLAMA=false 상태입니다.")

    primary = chains[0]
    if len(chains) > 1:
        primary = primary.with_fallbacks(chains[1:])

    _CACHED_CHAIN = primary
    _CACHED_CHAIN_LABEL = " -> ".join(labels)
    return _CACHED_CHAIN, _CACHED_CHAIN_LABEL




def get_existing_urls(urls: list) -> set:
    """여러 URL 중 이미 news_perspectives에 존재하는 URL 집합을 한 번의 쿼리로 반환합니다."""
    clean_urls = list({(u or "").strip() for u in (urls or []) if (u or "").strip()})
    if not clean_urls:
        return set()

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT url FROM news_perspectives WHERE url = ANY(%s);",
            (clean_urls,),
        )
        return {row[0] for row in cur.fetchall() if row and row[0]}
    finally:
        cur.close()
        conn.close()

def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    unescaped = html.unescape(str(text))
    return re.sub(r"\s+", " ", unescaped).strip()


def _parse_pub_date(raw_date: Optional[str], entry=None) -> datetime:
    if entry is not None:
        for attr in ("published_parsed", "updated_parsed"):
            parsed_tuple = getattr(entry, attr, None)
            if parsed_tuple:
                try:
                    return datetime(*parsed_tuple[:6])
                except Exception:
                    pass

    if raw_date:
        s = _clean_text(raw_date)
        s_no_tz = re.sub(r"\s*(KST|UTC|GMT|[+-]\d{2}:?\d{2})$", "", s, flags=re.IGNORECASE).strip()
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y.%m.%d %H:%M:%S",
            "%Y.%m.%d %H:%M",
        ):
            try:
                return datetime.strptime(s_no_tz, fmt)
            except ValueError:
                continue
        try:
            dt = parsedate_to_datetime(s)
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except Exception:
            pass

    return datetime.now()


def _fetch_xml_bytes(rss_url: str) -> Optional[bytes]:
    try:
        resp = requests.get(rss_url, headers=DEFAULT_HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"  [!] RSS HTTP 요청 실패 ({rss_url}): {e}", file=sys.stderr)
        return None


def parse_rss_items(rss_url: str) -> List[dict]:
    raw_xml = _fetch_xml_bytes(rss_url)
    items: List[dict] = []

    feed = feedparser.parse(raw_xml) if raw_xml else feedparser.parse(rss_url)
    for entry in (getattr(feed, "entries", []) or []):
        title = _clean_text(entry.get("title")) or "제목 없음"
        link = _clean_text(entry.get("link"))
        description = _clean_text(entry.get("summary") or entry.get("description"))
        author = _clean_text(entry.get("author"))
        raw_pub = _clean_text(entry.get("published") or entry.get("updated"))
        items.append({
            "title": title,
            "link": link,
            "description": description,
            "author": author,
            "pub_date_raw": raw_pub,
            "published_at": _parse_pub_date(raw_pub, entry=entry),
        })

    if raw_xml:
        try:
            root = ET.fromstring(raw_xml)
            xml_items = root.findall(".//item")
            if xml_items and (not items or len(xml_items) > len(items)):
                items = []
                for node in xml_items:
                    raw_pub = _clean_text(node.findtext("pubDate"))
                    items.append({
                        "title": _clean_text(node.findtext("title")) or "제목 없음",
                        "link": _clean_text(node.findtext("link")),
                        "description": _clean_text(node.findtext("description")),
                        "author": _clean_text(node.findtext("author")),
                        "pub_date_raw": raw_pub,
                        "published_at": _parse_pub_date(raw_pub),
                    })
            elif xml_items and items:
                for idx, node in enumerate(xml_items):
                    if idx >= len(items):
                        break
                    if not items[idx]["link"]:
                        items[idx]["link"] = _clean_text(node.findtext("link"))
                    if not items[idx]["author"]:
                        items[idx]["author"] = _clean_text(node.findtext("author"))
                    if not items[idx]["description"]:
                        items[idx]["description"] = _clean_text(node.findtext("description"))
                    if not items[idx]["pub_date_raw"]:
                        raw_pub = _clean_text(node.findtext("pubDate"))
                        items[idx]["pub_date_raw"] = raw_pub
                        items[idx]["published_at"] = _parse_pub_date(raw_pub)
        except Exception as e:
            print(f"  [!] XML ElementTree 보조 파싱 스킵: {e}", file=sys.stderr)

    return items


def _extract_html_with_bs4(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside"]):
        tag.decompose()

    for sel in (
        "#article-view-content-div",
        ".article-view-content",
        "#articleBody",
        ".article_body",
        ".news_cnt_detail_wrap",
        "article",
    ):
        el = soup.select_one(sel)
        if el:
            paragraphs = [
                _clean_text(p.get_text(" ", strip=True))
                for p in el.find_all(["p", "div"])
                if len(_clean_text(p.get_text(" ", strip=True))) >= 25
            ]
            joined = "\n".join(dict.fromkeys(paragraphs))
            if len(joined) >= 80:
                return joined

    return _clean_text(soup.get_text(" ", strip=True))


def fetch_article_details(url: str) -> dict:
    result = {"raw_content": "", "author": "", "thumbnail": "", "final_url": url}
    if not url:
        return result

    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=15, allow_redirects=True)
        result["final_url"] = response.url or url
        if response.status_code != 200:
            return result

        if not response.encoding or response.encoding.lower() in ("iso-8859-1", "windows-1252"):
            response.encoding = response.apparent_encoding or "utf-8"

        html_text = response.text
        extracted_json = trafilatura.extract(
            html_text,
            output_format="json",
            include_comments=False,
            include_tables=False,
            favor_precision=False,
        )
        if extracted_json:
            data = json.loads(extracted_json)
            result["raw_content"] = (data.get("text") or "").strip()
            result["author"] = _clean_text(data.get("author"))
            result["thumbnail"] = (data.get("image") or "").strip()

        if len(result["raw_content"]) < 80:
            bs_text = _extract_html_with_bs4(html_text)
            if len(bs_text) > len(result["raw_content"]):
                result["raw_content"] = bs_text

        return result
    except Exception as e:
        print(f"  [!] 본문 스크래핑 실패 ({url}): {e}", file=sys.stderr)
        return result


def _build_default_analysis(title: str, raw_content: str, category: str) -> dict:
    clean_text = (raw_content or title or "").strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?다])\s+", clean_text) if s.strip()]
    summary = " ".join(sentences[:3]) if sentences else clean_text[:300]
    return {
        "summary_facts": summary[:450],
        "author_opinion": "",
        "stance_type": "중립",
        "keywords": [category] if category else ["뉴스"],
        "model_used": "fallback_raw",
    }
def _get_groq_model_queue() -> List[str]:
    """환경변수에서 우선순위 순서대로 사용할 Groq 모델 목록을 반환합니다."""
    raw_list = os.getenv("GROQ_MODELS", "").strip()
    if raw_list:
        models = [m.strip() for m in raw_list.split(",") if m.strip()]
        if models:
            return list(dict.fromkeys(models))

    primary = (os.getenv("GROQ_MODEL") or "qwen/qwen3.8-27b").strip()
    fallbacks_raw = os.getenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b").strip()
    fallbacks = [m.strip() for m in fallbacks_raw.split(",") if m.strip()]
    return list(dict.fromkeys([primary] + fallbacks))

def _build_langchain_fallback_chain():
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "당신은 뉴스 기사 및 기자 관점 분석기입니다. 반드시 유효한 JSON 객체만 출력하세요. "
            "반드시 포함할 키: summary_facts (문자열), author_opinion (문자열), "
            "stance_type ('긍정','부정','비판','중립' 중 1개), keywords (문자열 리스트, 최대 5개).",
        ),
        (
            "human",
            "제목: {title}\n"
            "본문: {content}\n\n"
            "요구사항:\n"
            "1. summary_facts: 객관적인 핵심 사실 요약 (3문장 이내)\n"
            "2. author_opinion: 기자의 주관적 평가, 전망, 분석, 비판 ('기자는 ~평가한다' 같은 설명투 대신 핵심 논조만 간결히, 없으면 빈 문자열)\n"
            "3. stance_type: 긍정, 부정, 비판, 중립 중 하나\n"
            "4. keywords: 핵심 키워드 리스트 (최대 5개)\n"
            "{format_instructions}",
        ),
    ])

    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "700"))
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    ollama_model_name = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    chains = []
    labels = []

    if groq_api_key:
        for model_name in _get_groq_model_queue():
            groq_llm = ChatGroq(
                model=model_name,
                api_key=groq_api_key,
                temperature=0.1,
                max_tokens=groq_max_tokens,
            )
            groq_chain = (prompt | groq_llm | parser).with_config({"run_name": f"groq:{model_name}"})
            chains.append(groq_chain)
            labels.append(f"groq:{model_name}")

    if use_ollama:
        ollama_llm = ChatOllama(model=ollama_model_name, base_url=ollama_base_url, temperature=0.1)
        ollama_chain = (prompt | ollama_llm | parser).with_config({"run_name": f"ollama:{ollama_model_name}"})
        chains.append(ollama_chain)
        labels.append(f"ollama:{ollama_model_name}")

    if not chains:
        raise RuntimeError("GROQ_API_KEY가 설정되어 있지 않고 USE_OLLAMA=false 상태입니다.")

    primary = chains[0]
    if len(chains) > 1:
        primary = primary.with_fallbacks(chains[1:])

    return primary, " -> ".join(labels), None

def analyze_article_with_llm(title: str, raw_content: str, category: str = "뉴스") -> dict:
    default_res = _build_default_analysis(title, raw_content, category)
    if not raw_content or len(raw_content.strip()) < MIN_LLM_CONTENT_CHARS:
        return default_res

    text = re.sub(r"\s+", " ", raw_content).strip()
    max_input_chars = int(os.getenv("LLM_MAX_INPUT_CHARS", "1800"))
    if len(text) > max_input_chars:
        head_len = int(max_input_chars * 0.7)
        tail_len = max_input_chars - head_len
        content_for_llm = f"{text[:head_len]} ... {text[-tail_len:]}"
    else:
        content_for_llm = text

    try:
        chain, chain_label = _get_or_build_chain()
        global _LAST_USED_MODEL
        _LAST_USED_MODEL = None
        result = chain.invoke({
            "title": (title or "")[:120],
            "content": content_for_llm,
        })
        if isinstance(result, dict) and result.get("summary_facts"):
            keywords = result.get("keywords")
            if not isinstance(keywords, list) or not keywords:
                keywords = default_res["keywords"]
            return {
                "summary_facts": str(result.get("summary_facts")).strip(),
                "author_opinion": str(result.get("author_opinion") or "").strip(),
                "stance_type": str(result.get("stance_type") or "중립").strip(),
                "keywords": [str(k).strip() for k in keywords if str(k).strip()][:5],
                "model_used": _LAST_USED_MODEL or chain_label,
            }
    except Exception as e:
        print(f"  [!] LangChain 체인 상세 에러: {type(e).__name__}: {e}", file=sys.stderr)
        if hasattr(e, "exceptions"):
            for i, sub in enumerate(e.exceptions, 1):
                print(f"      └─ 폴백 #{i} 에러 ({type(sub).__name__}): {sub}", file=sys.stderr)

    return default_res

def _compose_raw_content(rss_description: str, scraped_content: str, fallback_title: str) -> str:
    desc = (rss_description or "").strip()
    body = (scraped_content or "").strip()
    if body and desc:
        probe = re.sub(r"\.{2,}$|…$", "", desc).strip()
        if probe and probe[:40] in body:
            return body
        return f"[RSS 요약] {desc}\n\n[본문]\n{body}"
    return body or desc or fallback_title


def _build_article_record(item: dict, scraped: dict, llm_result: dict, category: str, source_name: str, feed_url: str) -> dict:
    title = item["title"]
    link = item["link"]
    rss_summary = item["description"]
    raw_content = _compose_raw_content(rss_summary, scraped.get("raw_content", ""), title)
    author_name = item["author"] or scraped.get("author") or "기자 미상"
    final_url = scraped.get("final_url") or link or f"rss://{source_name}/{abs(hash(title))}"

    return {
        "title": title,
        "summary_facts": llm_result.get("summary_facts") or rss_summary or title,
        "author_opinion": llm_result.get("author_opinion") or "",
        "stance_type": llm_result.get("stance_type") or "중립",
        "category": category,
        "keywords": llm_result.get("keywords") or [category],
        "source": source_name,
        "url": final_url,
        "author_name": author_name,
        "published_at": item["published_at"],
        "thumbnail_url": scraped.get("thumbnail") or "",
        "raw_content": raw_content,
        "content_hash": generate_content_hash(raw_content) if raw_content else None,
        "model_version": llm_result.get("model_used", "groq:openai/gpt-oss-120b"),
        "metadata": {
            "collector": "rss_xml_bs4_trafilatura_langchain",
            "rss_feed_url": feed_url,
            "original_link": link,
            "pub_date_raw": item.get("pub_date_raw", ""),
            "rss_description": rss_summary,
            "scraped_success": bool(scraped.get("raw_content")),
        },
    }


def crawl_rss(
    rss_urls: Union[str, List[str]],
    category: str = "종합",
    source_name: str = "테스트언론사",
    limit: Optional[int] = 3,
    preview_chars: Optional[int] = 400,
    sim_threshold: float = 0.82,
) -> List[dict]:
    """[테스트 전용 - DB 저장 안 함] 이전 기사들과의 내용 비교기 결과 및 기자 연계 여부를 출력합니다."""
    urls = [rss_urls] if isinstance(rss_urls, str) else list(rss_urls)
    collected_samples: List[dict] = []
    seen_contents: List[tuple] = []

    print("\n" + "=" * 70, file=sys.stderr)
    print(f"🧪 [DRY-RUN 테스트 모드] DB 저장 없이 생데이터/비교기/가공결과 출력 ({source_name})", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    for feed_url in urls:
        if not feed_url:
            continue
        print(f"\n📡 RSS XML 파싱 중: {feed_url}", file=sys.stderr)
        items = parse_rss_items(feed_url)
        target_items = items[:limit] if limit is not None else items
        print(f"  └─ 전체 <item>: {len(items)}건 / 이번 테스트 대상: {len(target_items)}건", file=sys.stderr)

        for idx, item in enumerate(target_items, start=1):
            scraped = (
                fetch_article_details(item["link"])
                if item["link"]
                else {"raw_content": "", "author": "", "thumbnail": "", "final_url": ""}
            )
            raw_content = _compose_raw_content(item["description"], scraped.get("raw_content", ""), item["title"])
            llm_result = analyze_article_with_llm(item["title"], raw_content, category=category)
            record = _build_article_record(item, scraped, llm_result, category, source_name, feed_url)

            best_sim = 0.0
            matched_idx = None
            for prev_idx, prev_text in seen_contents:
                sim = compute_text_similarity(raw_content, prev_text)
                if sim > best_sim:
                    best_sim = sim
                    matched_idx = prev_idx

            is_dup_content = (matched_idx is not None and best_sim >= sim_threshold)
            if not is_dup_content:
                seen_contents.append((idx, raw_content))

            raw_preview = raw_content if preview_chars is None else raw_content[:preview_chars]
            raw_payload = {
                "rss_title": item["title"],
                "rss_link": item["link"],
                "final_url": record["url"],
                "rss_pub_date": item["pub_date_raw"],
                "parsed_published_at": record["published_at"].isoformat(),
                "author_name": record["author_name"],
                "thumbnail_url": record["thumbnail_url"],
                "scraped_success": record["metadata"]["scraped_success"],
                "content_hash": record["content_hash"],
                "content_comparator": {
                    "is_duplicate_content": is_dup_content,
                    "best_similarity_score": round(best_sim, 4),
                    "matched_article_index": matched_idx if is_dup_content else None,
                    "db_action": "link_journalist_only_skip_content" if is_dup_content else "insert_new_content_and_journalist",
                },
                "raw_content_length": len(raw_content),
                "raw_content_preview": raw_preview,
            }
            processed_payload = {
                "category": record["category"],
                "source": record["source"],
                "model_version": record["model_version"],
                "summary_facts": record["summary_facts"],
                "author_opinion": record["author_opinion"],
                "stance_type": record["stance_type"],
                "keywords": record["keywords"],
            }

            print("\n" + "-" * 70, file=sys.stderr)
            print(f"📰 [{idx}] {item['title']}", file=sys.stderr)
            print("-" * 70, file=sys.stderr)
            print("🔹 [1. REQUEST / XML / 스크래핑 생(RAW) 데이터]", file=sys.stderr)
            print(json.dumps(raw_payload, ensure_ascii=False, indent=2), file=sys.stderr)
            print("\n🔸 [2. LLM 가공(PROCESSED) 데이터]", file=sys.stderr)
            print(json.dumps(processed_payload, ensure_ascii=False, indent=2), file=sys.stderr)

            collected_samples.append({
                "raw": raw_payload,
                "processed": processed_payload,
                "record": record,
            })

    print("\n" + "=" * 70, file=sys.stderr)
    print(f"✅ [테스트 완료] 총 {len(collected_samples)}건 출력 (DB 미저장)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    return collected_samples


def crawl_and_save_rss(
    rss_urls: Union[str, List[str]],
    category: str,
    source_name: str,
    limit: Optional[int] = None,
    storage_mode: Optional[str] = None,
    sqlite_path: str = DEFAULT_SQLITE_PATH,
) -> int:
    """RSS URL들을 수집·분석하여 DB(postgres) 또는 파일(sqlite) 형태로 저장합니다."""
    mode = _resolve_storage_mode(storage_mode)
    urls = [rss_urls] if isinstance(rss_urls, str) else list(rss_urls)
    total_count = 0

    for feed_url in urls:
        if not feed_url:
            continue
        print(f"\n[{source_name}] RSS 수집 시작 ({mode}): {feed_url}", file=sys.stderr)
        all_items = parse_rss_items(feed_url)
        items = all_items[:limit] if (limit is not None and limit > 0) else all_items
        print(f"  └─ 발견된 <item> 수: {len(all_items)}건 / 처리 대상: {len(items)}건", file=sys.stderr )

        existing_urls = _fetch_existing_urls_by_mode(
            [it.get("link", "") for it in items],
            mode=mode,
            sqlite_path=sqlite_path,
        )
        seen_in_feed = set()

        for idx, item in enumerate(items, start=1):
            link = (item.get("link") or "").strip()
            if link and (link in existing_urls or link in seen_in_feed):
                print(f"  [{idx}/{len(items)}] 이미 처리된 링크 스킵: {item['title'][:35]}...", file=sys.stderr)
                continue
            if link:
                seen_in_feed.add(link)

            print(f"  [{idx}/{len(items)}] link 접속 중: {item['title'][:35]}...", file=sys.stderr)
            scraped = (
                fetch_article_details(link)
                if link
                else {"raw_content": "", "author": "", "thumbnail": "", "final_url": ""}
            )
            final_url = (scraped.get("final_url") or "").strip()
            if final_url and final_url != link and (final_url in existing_urls or final_url in seen_in_feed):
                print(f"  [{idx}/{len(items)}] 최종 URL 중복 스킵: {item['title'][:35]}...", file=sys.stderr)
                continue
            if final_url:
                seen_in_feed.add(final_url)

            raw_content = _compose_raw_content(item["description"], scraped.get("raw_content", ""), item["title"])
            llm_result = analyze_article_with_llm(item["title"], raw_content, category=category)
            article_data = _build_article_record(item, scraped, llm_result, category, source_name, feed_url)

            try:
                _persist_article_by_mode(article_data, mode=mode, sqlite_path=sqlite_path)
                total_count += 1
            except Exception as e:
                print(f"  [!] 저장 오류 발생, 백업 로그에 기록합니다: {e}", file=sys.stderr)
                _append_backup_record(article_data)
                total_count += 1

    print(f"[{source_name}] ({mode}) 총 {total_count}건 처리 완료!",file=sys.stderr)
    return total_count


def crawl_multiple_rss_sources(
    sources_config: List[dict],
    limit: Optional[int] = None,
    storage_mode: Optional[str] = None,
    sqlite_path: str = DEFAULT_SQLITE_PATH,
) -> int:
    grand_total = 0
    for cfg in sources_config:
        effective_limit = cfg.get("limit") if cfg.get("limit") is not None else limit
        effective_mode = cfg.get("storage_mode") or storage_mode
        effective_sqlite = cfg.get("sqlite_path") or sqlite_path
        grand_total += crawl_and_save_rss(
            rss_urls=cfg["rss_urls"],
            category=cfg.get("category", "종합"),
            source_name=cfg["source_name"],
            limit=effective_limit,
            storage_mode=effective_mode,
            sqlite_path=effective_sqlite,
        )
    return grand_total

def _append_backup_record(record: dict):
    try:
        backup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, "failed_db_inserts.jsonl")
        safe_record = {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in record.items()
        }
        with open(backup_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [!] 백업 파일 기록 실패: {e}", file=sys.stderr)


