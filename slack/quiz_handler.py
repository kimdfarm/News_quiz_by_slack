import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from QUIZ.quiz_engine import (
    QuizProjectDBMS,
    format_quiz_for_slack,
    generate_and_save_quiz_with_graph,
    regenerate_quiz_from_record,
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
        "당신은 1문항 퀴즈 답안 참/거짓 및 루브릭 채점관입니다. JSON으로만 답하세요.\n"
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


def _detect_regeneration_request(user_text: str, candidate_kw: Optional[str]) -> Optional[Dict[str, Any]]:
    """'~~프로젝트에서 문제 객관식/서술형으로 새롭게 만들어줘' 같은 재탄생 요청을 판별합니다."""
    topics = quiz_dbms.list_topics()
    if not topics:
        return None

    topic_names = [str(t.get("topic") or "") for t in topics if t.get("topic")]
    prompt = (
        "사용자 요청이 기존 PROJECT 문제집/프로젝트의 문제를 바탕으로 '새롭게/다시/재탄생/변형' 만들어 달라는 요청인지 판별하세요.\n"
        "JSON만 출력하세요.\n"
        '출력 형식: {"is_regen": true, "target_topic": "프로젝트명 또는 null", "target_format": "essay | multiple_choice | null"}\n\n'
        f"현재 PROJECT 주제 목록: {json.dumps(topic_names, ensure_ascii=False)}\n"
        f"후보 키워드: {candidate_kw!r}\n"
        f"사용자 원문: {user_text!r}"
    )
    parsed = extract_json(invoke_llm_with_fallback([HumanMessage(content=prompt)], temperature=0.0, max_tokens=150, prefer_local=True) or "")
    if not parsed or not parsed.get("is_regen"):
        return None

    raw_topic = str(parsed.get("target_topic") or candidate_kw or "").strip()
    matched_topic = None
    if raw_topic:
        for t_name in topic_names:
            if t_name.lower() == raw_topic.lower() or raw_topic.lower() in t_name.lower() or t_name.lower() in raw_topic.lower():
                matched_topic = t_name
                break
        if not matched_topic:
            matched_topic = raw_topic

    records = quiz_dbms.select_quizzes(topic=matched_topic) if matched_topic else quiz_dbms.select_quizzes()
    if not records:
        return None

    fmt = str(parsed.get("target_format") or "").strip().lower()
    if fmt not in ("essay", "multiple_choice"):
        fmt = None

    return {
        "source_quiz": records[-1],
        "target_format": fmt,
    }


def _post_and_bind_thread(result: Dict[str, Any], event: Dict[str, Any], say: Any) -> None:
    posted = say(
        text=format_quiz_for_slack(
            result,
            articles=result.get("articles"),
            show_sources=False,
            show_rubric=False,
        )
    )
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


def handle_quiz_generate(
    event: Dict[str, Any],
    user_text: str,
    chosen: Dict[str, Any],
    say: Any,
    mcp_tools: Any,
) -> None:
    merged = unwrap_tool_args(chosen)
    candidate_kw = clean_str_arg(merged.get("keyword")) or clean_str_arg(merged.get("category"))
    q_type = clean_str_arg(merged.get("question_type"))

    regen_info = _detect_regeneration_request(user_text, candidate_kw)
    if regen_info:
        say(text="♻️ 기존 프로젝트 문제를 바탕으로 *독립된 새 문제*로 재탄생시키는 중이에요...")
        regen_result = regenerate_quiz_from_record(
            source_quiz=regen_info["source_quiz"],
            user_text=user_text,
            llm_call_fn=_llm_for_quiz_graph,
            dbms=quiz_dbms,
            target_format=regen_info["target_format"],
            question_type=q_type,
        )
        _post_and_bind_thread(regen_result, event=event, say=say)
        return

    raw_kw = resolve_quiz_keyword_with_llm(candidate_kw, user_text=user_text)
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

    say(text="🧠 Top-K 도구 라우팅 및 *LangGraph + LangSmith*로 1문항 퀴즈를 생성해 `PROJECT/`에 저장 중이에요...")
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
    _post_and_bind_thread(result, event=event, say=say)


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
    
from QUIZ.quiz_engine import CIRCLE_NUMS, parse_n_m_from_text


def _extract_choice_numbers(user_answer: str, max_choices: int) -> List[int]:
    s = (user_answer or "").strip()
    found: List[int] = []
    for idx, sym in enumerate(CIRCLE_NUMS[:max_choices], 1):
        if sym in s and idx not in found:
            found.append(idx)

    for m in re.finditer(r"(?:^|[^\d])([1-9]|10)(?:\s*번|[^\d]|$)", s):
        num = int(m.group(1))
        if 1 <= num <= max_choices and num not in found:
            found.append(num)

    return sorted(found)


def grade_multiple_choice_deterministic(
    quiz_record: Dict[str, Any],
    selected_indices: Optional[List[int]],
    raw_user_text: str,
    user_id: str,
) -> str:
    q = (quiz_record.get("questions") or [{}])[0]
    choices = q.get("choices") or []
    n_cnt = len(choices)
    raw_ans_indices = q.get("answer_indices")
    if isinstance(raw_ans_indices, list) and raw_ans_indices:
        ans_indices = sorted({int(x) for x in raw_ans_indices if 1 <= int(x) <= n_cnt})
    elif q.get("answer_index"):
        ans_indices = [int(q["answer_index"])]
    else:
        ans_indices = [1] if n_cnt >= 1 else []

    expected_m = int(q.get("num_correct") or len(ans_indices) or 1)
    chosen_list = (
        sorted({int(x) for x in selected_indices if 1 <= int(x) <= n_cnt})
        if selected_indices is not None
        else _extract_choice_numbers(raw_user_text, max_choices=n_cnt)
    )

    if not chosen_list:
        return (
            f"<@{user_id}> 선택하신 번호를 인식하지 못했어요. "
            f"`1번`~`{n_cnt}번`({expected_m}개)으로 입력하거나 위 UI에서 선택 후 제출해 주세요."
        )

    chosen_labels = ", ".join(choices[i - 1] for i in chosen_list)
    ans_labels = ", ".join(choices[i - 1] for i in ans_indices)
    is_correct = (chosen_list == ans_indices)

    if is_correct:
        verdict = "⭕ *정답입니다!*"
    elif len(chosen_list) != expected_m:
        verdict = f"❌ *오답입니다.* (이 문항은 *{expected_m}개* 정답인데 *{len(chosen_list)}개*를 선택하셨습니다.)"
    else:
        verdict = "❌ *오답입니다.*"

    lines = [
        f"<@{user_id}> 님의 답을 *{chosen_labels}* (으)로 제출했습니다.",
        f"• 판정: {verdict}",
        f"• 정답({len(ans_indices)}개): *{ans_labels}*",
    ]
    if q.get("explanation"):
        lines.append(f"• 해설: {q['explanation']}")
    return "\n".join(lines)    
    
def handle_quiz_ui_submit(body: Dict[str, Any], client: Any) -> None:
    user_id = (body.get("user") or {}).get("id") or "unknown"
    channel_id = (body.get("channel") or {}).get("id")
    message = body.get("message") or {}
    thread_ts = message.get("thread_ts") or message.get("ts")

    actions = body.get("actions") or []
    quiz_id = actions[0].get("value") if actions else None

    state_values = ((body.get("state") or {}).get("values") or {})
    block_key = f"quiz_select_block_{quiz_id}" if quiz_id else ""
    action_obj = state_values.get(block_key, {}).get("quiz_choice_select") or {}

    selected_values: List[str] = []
    if isinstance(action_obj.get("selected_options"), list):
        for opt in action_obj["selected_options"]:
            if isinstance(opt, dict) and opt.get("value"):
                selected_values.append(str(opt["value"]))
    elif isinstance(action_obj.get("selected_option"), dict) and action_obj["selected_option"].get("value"):
        selected_values.append(str(action_obj["selected_option"]["value"]))

    matched_quiz = None
    if quiz_id:
        found = quiz_dbms.select_quizzes(quiz_id=quiz_id)
        matched_quiz = found[0] if found else None
    if not matched_quiz and thread_ts:
        matched_quiz = quiz_dbms.get_by_thread(str(thread_ts))

    if not matched_quiz or not channel_id or not thread_ts:
        return

    if not selected_values:
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=str(thread_ts),
            text=f"<@{user_id}> 먼저 위 선택지를 고른 뒤 *답안 제출하기*를 눌러주세요.",
        )
        return

    selected_indices: List[int] = []
    for val in selected_values:
        head = val.split("|", 1)[0]
        if head.isdigit():
            selected_indices.append(int(head))

    reply_text = grade_multiple_choice_deterministic(
        quiz_record=matched_quiz,
        selected_indices=selected_indices,
        raw_user_text=", ".join(selected_values),
        user_id=user_id,
    )
    client.chat_postMessage(
        channel=channel_id,
        thread_ts=str(thread_ts),
        text=reply_text,
    )














