import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from QUIZ.quiz_engine import (
    QuizProjectDBMS,
    format_quiz_for_slack,
    generate_and_save_quiz_with_graph,
)
from slack.llm_router import (
    clean_str_arg,
    extract_json,
    invoke_llm_with_fallback,
    resolve_positive_int_with_llm,
    resolve_quiz_keyword_with_llm,
    unwrap_tool_args,
)
from storage_backend import DEFAULT_SQLITE_PATH

_PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PROJECT")
quiz_dbms = QuizProjectDBMS(base_dir=_PROJECT_DIR)


def fetch_articles_for_quiz(keyword: Optional[str], limit: int, mcp_tools: Any) -> List[Dict[str, Any]]:
    fetch_n = max(limit * 3, 15)
    raw_res = mcp_tools.get_recent_news_for_quiz(
        category="전체",
        limit=fetch_n,
        sqlite_path=DEFAULT_SQLITE_PATH,
    )
    bundle = json.loads(raw_res) if raw_res else {}
    articles = list(bundle.get("articles") or [])

    if keyword:
        cat_res = mcp_tools.get_recent_news_for_quiz(
            category=keyword,
            limit=fetch_n,
            sqlite_path=DEFAULT_SQLITE_PATH,
        )
        cat_bundle = json.loads(cat_res) if cat_res else {}
        seen_titles = {a.get("title") for a in articles}
        for a in cat_bundle.get("articles") or []:
            if a.get("title") not in seen_titles:
                articles.append(a)
                seen_titles.add(a.get("title"))

        kw_low = keyword.lower()
        matched = [
            a
            for a in articles
            if kw_low in str(a.get("title") or "").lower()
            or kw_low in str(a.get("facts") or "").lower()
            or kw_low in str(a.get("opinion") or "").lower()
            or kw_low in str(a.get("category") or "").lower()
        ]
        if matched:
            return matched[:limit]

    return articles[:limit]


def _llm_for_quiz_graph(prompt: str, max_tokens: int) -> Optional[str]:
    prefer_local = (os.getenv("QUIZ_LLM_PROVIDER") or "").strip().lower() == "ollama"
    return invoke_llm_with_fallback(
        messages=[HumanMessage(content=prompt)],
        temperature=0.3,
        max_tokens=max_tokens,
        prefer_local=prefer_local,
    )


def _grade_thread_submission(quiz_record: Dict[str, Any], user_answer: str, user_id: str) -> str:
    ls_project = (quiz_record.get("langsmith") or {}).get("project_name")
    if ls_project:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = ls_project

    prompt = (
        "당신은 퀴즈 답안 참/거짓 및 루브릭 채점관입니다. JSON으로만 답하세요.\n"
        f"문제 정보: {json.dumps(quiz_record, ensure_ascii=False)[:2500]}\n"
        f"제출자(<@{user_id}>) 답안: {user_answer}\n\n"
        '출력 형식: {"is_true": true, "score_text": "정답/충족", "reason": "판별 이유", "missing": []}'
    )
    raw = (
        invoke_llm_with_fallback(
            messages=[HumanMessage(content=prompt)],
            temperature=0.1,
            max_tokens=500,
            prefer_local=False,
        )
        or ""
    )
    parsed = extract_json(raw)
    if not parsed:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        parsed = json.loads(m.group(0)) if m else {"is_true": False, "reason": raw.strip()}

    verdict_label = "참 (정답/기준 충족)" if parsed.get("is_true") else "거짓 (오답/기준 미충족)"
    lines = [
        f"<@{user_id}> 답안 판별: *{verdict_label}* ({parsed.get('score_text', '')})",
        f"• 근거: {parsed.get('reason', '')}",
    ]
    if parsed.get("missing"):
        lines.append(f"• 보완할 점: {', '.join(str(x) for x in parsed['missing'])}")
    return "\n".join(lines)

def handle_quiz_generate(
    event: Dict[str, Any],
    user_text: str,
    chosen: Dict[str, Any],
    say: Any,
    mcp_tools: Any,
) -> None:
    merged = unwrap_tool_args(chosen)
    candidate_kw = clean_str_arg(merged.get("keyword")) or clean_str_arg(merged.get("category"))
    raw_kw = resolve_quiz_keyword_with_llm(candidate_kw, user_text=user_text)
    q_type = clean_str_arg(merged.get("question_type"))
    llm_rule = clean_str_arg(merged.get("llm_rule"))
    raw_rubric = merged.get("custom_rubric")
    rubric_list = [str(x) for x in raw_rubric] if isinstance(raw_rubric, list) else None
    limit = (
        resolve_positive_int_with_llm(
            merged.get("limit"),
            user_text=user_text,
            field_name="limit",
            default=4,
        )
        or 4
    )

    say(text="🧠 Top-K 도구 라우팅 및 *LangGraph + LangSmith*로 퀴즈를 생성해 `PROJECT/`에 저장 중이에요...")
    result = generate_and_save_quiz_with_graph(
        user_text=user_text,
        fetch_articles_fn=lambda kw, lim: fetch_articles_for_quiz(kw, lim, mcp_tools=mcp_tools),
        llm_call_fn=_llm_for_quiz_graph,
        dbms=quiz_dbms,
        keyword=raw_kw,
        question_type=q_type,
        custom_rubric=rubric_list,
        llm_rule=llm_rule,
        limit=limit,
    )
    posted = say(text=format_quiz_for_slack(result, articles=result.get("articles")))
    posted_data = getattr(posted, "data", posted) if posted is not None else {}
    thread_ts = None
    if isinstance(posted_data, dict):
        thread_ts = posted_data.get("ts") or (posted_data.get("message") or {}).get("ts")
    if thread_ts:
        rec = result["quiz_record"]
        quiz_dbms.save_quiz(
            topic=rec["topic"],
            quiz_record=rec,
            thread_ts=str(thread_ts),
            channel_id=event.get("channel"),
        )


def handle_quiz_db_manage(event: Dict[str, Any], chosen: Dict[str, Any], say: Any, client: Any) -> None:
    merged = unwrap_tool_args(chosen)
    action = (clean_str_arg(merged.get("action")) or "list").lower()
    topic = clean_str_arg(merged.get("topic"))
    quiz_id = clean_str_arg(merged.get("quiz_id"))
    style = clean_str_arg(merged.get("template_style")) or "csat_2col"

    if action == "list":
        topics = quiz_dbms.list_topics()
        if not topics:
            say(text="📂 `PROJECT/` 폴더에 저장된 문제 파일이 아직 없어요.")
            return
        lines = [f"• `{t['file']}` (주제: *{t['topic']}*, {t['count']}문항)" for t in topics]
        say(text="📂 *PROJECT 퀴즈 DB 목록*\n" + "\n".join(lines))
        return

    if action == "export":
        out_path = os.path.join(_PROJECT_DIR, f"bundle_{style}.json")
        saved = quiz_dbms.export_bundle(
            export_path=out_path,
            topics=[topic] if topic else None,
            template_style=style,
        )
        client.files_upload_v2(
            channel=event.get("channel"),
            file=saved,
            filename=os.path.basename(saved),
            title=os.path.basename(saved),
            initial_comment=f"📦 문제집 번들(`{style}`)을 디코딩해 추출했어요.",
        )
        return

    if action == "delete" and quiz_id:
        deleted = quiz_dbms.delete_quiz(quiz_id)
        say(text=f"🗑️ `{quiz_id}` 삭제 {'완료' if deleted else '대상 없음'}.")
        return

    items = quiz_dbms.select_quizzes(topic=topic, quiz_id=quiz_id)
    say(text=f"🔎 조회 결과 `{len(items)}`건 (`topic={topic or '전체'}`)")
    
def _format_requested_quiz_extras(quiz_record: Dict[str, Any], req_text: str) -> Optional[str]:
    low = req_text.lower()
    want_rubric = any(k in low for k in ("채점 기준", "채점기준", "루브릭", "기준 보여", "기준 알려", "기준도"))
    want_sources = any(k in low for k in ("참고 기사", "참고자료", "참고 자료", "기사 보여", "기사 알려", "출처", "참고도"))

    if not want_rubric and not want_sources:
        return None

    parts: List[str] = []
    if want_sources:
        disp_limit = int(quiz_record.get("source_display_limit") or len(quiz_record.get("source_excerpts") or []) or 1)
        excerpts = (quiz_record.get("source_excerpts") or [])[:disp_limit]
        if excerpts:
            parts.append("*참고 기사 자료*")
            for ex in excerpts:
                parts.append(f"• *{ex.get('source') or '기사'}*: {ex.get('summary') or ''}")
        else:
            parts.append("*참고 기사 자료*\n• 저장된 기사 발췌가 없습니다.")

    if want_rubric:
        rubric = quiz_record.get("rubric") or []
        parts.append("*채점 기준*")
        if rubric:
            for r in rubric:
                parts.append(f"• {r}")
        else:
            parts.append("• 등록된 채점 기준이 없습니다.")

    return "\n".join(parts)


def try_handle_quiz_thread_reply(thread_ts: str, answer_text: str, user_id: str, say: Any) -> bool:
    if not thread_ts or not answer_text:
        return False
    matched_quiz = quiz_dbms.get_by_thread(str(thread_ts))
    if not matched_quiz:
        return False

    extras_msg = _format_requested_quiz_extras(matched_quiz, answer_text)
    if extras_msg:
        say(text=extras_msg, thread_ts=thread_ts)
        return True

    verdict_msg = _grade_thread_submission(
        quiz_record=matched_quiz,
        user_answer=answer_text,
        user_id=user_id or "unknown",
    )
    say(text=verdict_msg, thread_ts=thread_ts)
    return True




