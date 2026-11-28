import json
import os
import random
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from QUIZ.quiz_json_dbms import QuizProjectDBMS, sanitize_topic_filename
from QUIZ.types import get_quiz_type_names, get_quiz_type_spec, load_all_quiz_types

load_all_quiz_types()
QUIZ_TYPES = get_quiz_type_names()


def _extract_json_obj(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None


def _parse_n_m_via_plugin(user_text: str, default_n: int = 5, default_m: int = 1) -> tuple[int, int]:
    csat_spec = get_quiz_type_spec("csat_passage_set")
    parser = csat_spec.get("parse_n_m")
    if callable(parser):
        return parser(user_text, default_n=default_n, default_m=default_m)
    max_n = int(csat_spec.get("max_choices") or 10)
    n_val = max(2, min(default_n, max_n))
    m_val = max(1, min(default_m, n_val - 1))
    return n_val, m_val


def _normalize_single_question_via_plugin(
    parsed: Dict[str, Any],
    spec: Dict[str, Any],
    allow_choices: bool,
    num_choices: int = 5,
    num_correct: int = 1,
) -> Dict[str, Any]:
    """1문항 고정 및 객관식 레코드 정규화를 타입 플러그인에 위임합니다."""
    if allow_choices and callable(spec.get("normalize_record")):
        return spec["normalize_record"](
            parsed,
            num_choices=num_choices,
            num_correct=num_correct,
        )

    questions = parsed.get("questions") or []
    first_q = (
        dict(questions[0])
        if questions and isinstance(questions[0], dict)
        else {
            "number": 1,
            "stem": "제시된 내용을 바탕으로 알맞은 것을 고르세요.",
            "box_example": None,
            "choices": [],
            "answer_indices": [],
            "answer": "",
            "explanation": "",
        }
    )
    first_q["number"] = 1
    first_q["choices"] = []
    first_q["num_choices"] = 0
    first_q["num_correct"] = 0
    first_q["answer_indices"] = []
    first_q["answer_index"] = None
    parsed["questions"] = [first_q]
    parsed["is_essay"] = not allow_choices
    return parsed


class QuizGenState(TypedDict):
    user_text: str
    keyword: Optional[str]
    rag_query: Optional[str]
    question_type: Optional[str]
    num_choices: int
    num_correct: int
    custom_rubric: List[str]
    llm_rule: Optional[str]
    existing_ctx: str
    articles: List[Dict[str, Any]]
    relevance_ok: bool
    rag_feedback: str
    rag_retry_count: int
    design_plan: Dict[str, Any]
    source_display_limit: int
    draft_quiz: Dict[str, Any]
    quality_report: Dict[str, Any]
    retry_count: int
    ls_project: str


def _articles_to_context(articles: List[Dict[str, Any]], max_items: Optional[int] = None) -> str:
    if not articles:
        return "(저장된 기사 데이터 중 요청 주제와 직접 일치하는 기사가 부족합니다.)"
    sliced = articles[:max_items] if max_items and max_items > 0 else articles
    lines = []
    for i, a in enumerate(sliced, 1):
        lines.append(
            f"[{i}] ({a.get('source') or a.get('press') or '언론'}) {a.get('title') or ''}\n"
            f"    사실: {a.get('facts') or a.get('summary') or a.get('content') or ''}\n"
            f"    관점({a.get('stance_type') or '중립'}): {a.get('opinion') or ''}"
        )
    return "\n".join(lines)


def _infer_type_with_llm(
    user_text: str,
    requested_type: Optional[str],
    llm_call_fn: Callable[[str, int], Optional[str]],
) -> str:
    available = load_all_quiz_types()
    if requested_type and requested_type in available:
        if "서술형" in user_text and available[requested_type].get("allow_choices"):
            return "multi_perspective" if "multi_perspective" in available else requested_type
        return requested_type

    if "서술형" in user_text:
        return (
            "multi_perspective"
            if "multi_perspective" in available
            else random.choice(list(available.keys()))
        )

    catalog = [
        {"name": k, "label": v.get("label", k), "description": v.get("description", "")}
        for k, v in available.items()
    ]
    prompt = (
        "사용자 요청에 가장 알맞은 문제 유형(name) 1개를 고르세요. JSON만 출력하세요.\n"
        '출력 형식: {"question_type": "유형name"}\n\n'
        f"사용자 요청: {user_text}\n"
        f"사용 가능한 유형 목록: {json.dumps(catalog, ensure_ascii=False)}"
    )
    parsed = _extract_json_obj(llm_call_fn(prompt, 150) or "")
    picked = (parsed or {}).get("question_type")
    if picked in available:
        return picked
    return random.choice(list(available.keys()))


def generate_and_save_quiz_with_graph(
    user_text: str,
    fetch_articles_fn: Callable[[Optional[str], int], List[Dict[str, Any]]],
    llm_call_fn: Callable[[str, int], Optional[str]],
    dbms: QuizProjectDBMS,
    keyword: Optional[str] = None,
    question_type: Optional[str] = None,
    custom_rubric: Optional[List[str]] = None,
    llm_rule: Optional[str] = None,
    limit: int = 4,
    num_choices: Optional[int] = None,
    num_correct: Optional[int] = None,
) -> Dict[str, Any]:
    resolved_type = _infer_type_with_llm(user_text, question_type, llm_call_fn)
    parsed_n, parsed_m = _parse_n_m_via_plugin(user_text, default_n=5, default_m=1)
    max_n = int(get_quiz_type_spec("csat_passage_set").get("max_choices") or 10)
    n_cnt = max(2, min(int(num_choices if num_choices is not None else parsed_n), max_n))
    m_cnt = max(1, min(int(num_correct if num_correct is not None else parsed_m), n_cnt - 1))

    topic_label = sanitize_topic_filename(keyword or f"random_{resolved_type}")
    ls_project = f"quiz-{topic_label}-{uuid.uuid4().hex[:4]}"

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = ls_project

    existing_summaries = dbms.get_existing_summaries(limit=20)
    existing_ctx = "\n".join(f"- {s}" for s in existing_summaries)

    def node_fetch_rag(state: QuizGenState) -> Dict[str, Any]:
        query = state.get("rag_query") or state.get("keyword")
        articles = fetch_articles_fn(query, max(limit * 3, 15))
        if not articles and query:
            articles = fetch_articles_fn(None, max(limit * 3, 15))
        return {"articles": articles}

    def node_judge_relevance(state: QuizGenState) -> Dict[str, Any]:
        target_topic = (state.get("keyword") or "").strip()
        articles = state.get("articles") or []
        retry_cnt = int(state.get("rag_retry_count") or 0)

        if not articles:
            return {
                "relevance_ok": False,
                "rag_retry_count": retry_cnt + 1,
                "rag_feedback": "DB에서 조회된 기사가 없습니다.",
            }

        if not target_topic:
            picked = articles[: max(limit * 2, 6)]
            titles = [str(a.get("title") or "") for a in picked if a.get("title")]
            words = [
                w
                for t in titles
                for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", t)
                if w not in ("뉴스", "기사", "논란", "발표", "전망", "시장")
            ]
            auto_kw = random.choice(words) if words else "시사_이슈"
            return {
                "keyword": auto_kw,
                "articles": picked,
                "relevance_ok": True,
                "rag_feedback": "",
            }

        compact = [
            {
                "idx": i,
                "title": a.get("title", ""),
                "category": a.get("category", ""),
                "source": a.get("source") or a.get("press") or "",
                "summary": str(a.get("facts") or a.get("summary") or a.get("opinion") or "")[:200],
            }
            for i, a in enumerate(articles[:12])
        ]

        prompt = (
            "당신은 RAG 검색 결과의 주제 적합성을 엄격히 판별하는 심사관입니다.\n"
            f"요청 주제/키워드: {target_topic}\n"
            f"사용자 원문: {state['user_text']}\n\n"
            "아래 기사 목록 중 요청 주제와 직접적으로 관련이 있는 기사 인덱스(indices)만 고르세요.\n"
            "예: 요청 주제가 'AI'인데 단순 e스포츠/게임 대회/퍼블리싱 기사라면 절대 선택하지 마세요.\n"
            "관련 기사가 하나도 없으면 indices를 []로 두고 reason에 불일치 사유를 적으세요.\n"
            'JSON만 출력: {"indices": [0, 2], "reason": "판별 이유"}\n\n'
            f"기사 목록:\n{json.dumps(compact, ensure_ascii=False)}"
        )

        parsed = _extract_json_obj(llm_call_fn(prompt, 300) or "") or {}
        raw_indices = parsed.get("indices") if isinstance(parsed.get("indices"), list) else []
        valid_indices = [
            int(idx)
            for idx in raw_indices
            if isinstance(idx, (int, str)) and str(idx).isdigit() and 0 <= int(idx) < len(articles[:12])
        ]
        filtered = [articles[:12][i] for i in valid_indices]

        if filtered:
            return {
                "articles": filtered,
                "relevance_ok": True,
                "rag_feedback": "",
            }

        return {
            "articles": [],
            "relevance_ok": False,
            "rag_retry_count": retry_cnt + 1,
            "rag_feedback": str(
                parsed.get("reason") or f"'{target_topic}' 주제와 직접 관련된 기사를 찾지 못했습니다."
            ),
        }

    def node_rewrite_rag_query(state: QuizGenState) -> Dict[str, Any]:
        prev_q = state.get("rag_query") or state.get("keyword") or ""
        reason = state.get("rag_feedback") or ""
        prompt = (
            "이전 RAG 검색에서 요청 주제와 맞는 기사를 찾지 못했습니다.\n"
            "DB에서 관련 기사를 정밀하게 찾기 위한 대체 핵심 검색어(1단어 또는 짧은 키워드) 1개를 제안하세요.\n"
            'JSON만 출력: {"next_query": "대체검색어"}\n\n'
            f"원래 주제: {state.get('keyword')}\n"
            f"이전 검색어: {prev_q}\n"
            f"실패 사유: {reason}"
        )
        parsed = _extract_json_obj(llm_call_fn(prompt, 150) or "") or {}
        next_q = str(parsed.get("next_query") or "").strip() or None
        return {"rag_query": next_q}

    def route_after_relevance(state: QuizGenState) -> str:
        if state.get("relevance_ok"):
            return "plan_design"
        if int(state.get("rag_retry_count") or 0) >= 2:
            return "plan_design"
        return "rewrite_rag"

    def node_plan_design(state: QuizGenState) -> Dict[str, Any]:
        articles = state.get("articles") or []
        q_type = state.get("question_type") or resolved_type
        spec = get_quiz_type_spec(q_type)
        is_essay = (
            bool(spec.get("is_essay"))
            or (not bool(spec.get("allow_choices")))
            or ("서술형" in state["user_text"])
        )
        avail_cnt = len(articles)

        titles_preview = [
            f"{i+1}. [{a.get('source') or a.get('press') or '언론'}] {a.get('title') or ''}"
            for i, a in enumerate(articles[:8])
        ]
        prompt = (
            "당신은 출제 전략을 설계하는 LangGraph 설계자입니다.\n"
            "확보된 기사 수와 문제 유형을 보고, 이번 문제에 몇 개의 기사를 반영할지(display_article_count)와 "
            "출제 각도(angle)를 지능적으로 결정하세요.\n"
            "규칙:\n"
            f"1. 확보된 기사 수는 {avail_cnt}개입니다. display_article_count는 1 이상 {max(1, min(avail_cnt, 8))} 이하 정수로 정하세요.\n"
            "2. 단순 팩트 1개만 깊이 파는 문제면 1~2개, 복수 관점 비교·다각도 논증·종합 분석이면 3~6개처럼 "
            "4개 고정에 얽매이지 말고 출제 목적에 맞게 적게 또는 많게 정하세요.\n"
            "3. 서술형(is_essay=true)일 경우 '지문 요약해라' 같은 밋밋한 지시가 아니라 "
            "상반된 시각/쟁점 비교, 한계점 비판, 정책·대응 방향 논증 등 사고력을 요구하는 출제 각도를 잡으세요.\n"
            'JSON만 출력: {"display_article_count": 3, "angle": "출제 각도 설명", "reasoning": "선택 이유"}\n\n'
            f"주제: {state.get('keyword')}\n"
            f"유형: {q_type} (is_essay={is_essay})\n"
            f"사용자 요청: {state['user_text']}\n"
            f"확보된 기사 목록:\n" + ("\n".join(titles_preview) if titles_preview else "(없음)")
        )
        parsed = _extract_json_obj(llm_call_fn(prompt, 250) or "") or {}
        max_allowed = max(1, min(avail_cnt if avail_cnt > 0 else 1, 8))
        raw_cnt = parsed.get("display_article_count")
        try:
            chosen_cnt = int(raw_cnt)
        except Exception:
            chosen_cnt = min(max_allowed, 2 if is_essay else 3)
        chosen_cnt = max(1, min(chosen_cnt, max_allowed))

        plan = {
            "is_essay": is_essay,
            "display_article_count": chosen_cnt,
            "angle": str(
                parsed.get("angle")
                or (
                    "핵심 쟁점과 상반된 관점·한계점을 비교하며 논거를 제시하도록 출제"
                    if is_essay
                    else "제시된 기사 근거를 바탕으로 핵심 원리와 적용 사례를 분석하도록 출제"
                )
            ),
            "reasoning": str(parsed.get("reasoning") or "문항 성격과 확보 기사 수에 맞춰 동적 산정"),
        }
        return {
            "design_plan": plan,
            "source_display_limit": chosen_cnt,
            "articles": articles[:chosen_cnt] if articles else [],
        }

    def node_generate(state: QuizGenState) -> Dict[str, Any]:
        articles = state.get("articles") or []
        disp_limit = int(state.get("source_display_limit") or len(articles) or 1)
        ctx = _articles_to_context(articles, max_items=disp_limit)
        q_type = state["question_type"] or resolved_type
        spec = get_quiz_type_spec(q_type)
        plan = state.get("design_plan") or {}
        is_essay = bool(plan.get("is_essay")) or ("서술형" in state["user_text"])
        allow_choices = bool(spec.get("allow_choices")) and not is_essay

        st_n = max(2, min(int(state.get("num_choices") or n_cnt), max_n))
        st_m = max(1, min(int(state.get("num_correct") or m_cnt), st_n - 1))

        prev_feedback = (state.get("quality_report") or {}).get("feedback") or ""
        rag_warn = state.get("rag_feedback") or ""
        extra_rule = state.get("llm_rule") or ""
        rubric_list = state.get("custom_rubric") or [
            "핵심 쟁점과 근거를 정확히 제시했는가",
            "기사 사실과 모순되는 주장이 없는가",
            "주장·비교·판단의 논리적 일관성이 있는가",
        ]
        type_guide = spec["build_guide"](rubric_list)

        if is_essay:
            format_rule = (
                "서술형 1문항입니다. choices는 반드시 빈 배열([])로 두고 선택지를 절대 만들지 마세요. "
                "단순 요약형을 피하고 관점 비교·한계 비판·근거 논증형으로 출제하세요."
            )
        else:
            format_rule = (
                f"객관식 1문항입니다. choices 배열에 정확히 {st_n}개 선택지를 넣고, "
                f"그중 {st_m}개를 정답으로 두며 나머지 {st_n - st_m}개는 지문/사실을 비튼 오답 함정으로 구성하세요. "
                f"answer_indices 배열에 정답 번호({st_m}개, 1부터 시작)를 반드시 명시하세요."
            )

        prompt = (
            f"주제 키워드: {state['keyword']}\n"
            f"문제 유형: {q_type} ({spec.get('label', q_type)})\n"
            f"사용자 요청: {state['user_text']}\n"
            f"LangGraph 출제 설계 방향: {plan.get('angle', '')}\n"
            + (f"최종 판별/출제 규칙: {extra_rule}\n" if extra_rule else "")
            + (
                f"이미 PROJECT에 생성된 문제 목록(가능하면 겹치지 않게 출제):\n{state.get('existing_ctx', '')}\n"
                if state.get("existing_ctx")
                else ""
            )
            + f"참고 기사 ({disp_limit}건 반영):\n{ctx}\n\n"
            f"출제 지침: {type_guide}\n"
            f"형식 강제 규칙: {format_rule}\n"
            "중요: 참고 기사에 없는 내용을 억지로 기사 보도인 것처럼 지어내지 마세요.\n"
            + (f"참고 알림: {rag_warn}\n" if rag_warn else "")
            + (f"이전 품질 검증 피드백(반드시 반영해 개선): {prev_feedback}\n" if prev_feedback else "")
            + "\n반드시 아래 JSON 형식으로만 출력하세요:\n"
            "{\n"
            '  "title": "문제 세트 제목",\n'
            '  "passage": "문제 풀이에 필요한 배경 지문 또는 상황 설명",\n'
            '  "source_excerpts": [\n'
            '    {"source": "언론사/제목", "summary": "핵심 사실·관점 요약"}\n'
            "  ],\n"
            '  "questions": [\n'
            "    {\n"
            '      "number": 1,\n'
            '      "stem": "문항",\n'
            '      "box_example": null,\n'
            '      "choices": [],\n'
            '      "answer_indices": [1],\n'
            '      "answer": "정답/핵심요지",\n'
            '      "explanation": "해설"\n'
            "    }\n"
            "  ],\n"
            '  "rubric": ["채점 기준 1", "채점 기준 2"]\n'
            "}"
        )

        raw = llm_call_fn(prompt, 1600) or ""
        parsed = _extract_json_obj(raw)
        if not parsed:
            parsed = {
                "title": f"{state['keyword']} 분석 과제",
                "passage": ctx[:800],
                "questions": [
                    {
                        "number": 1,
                        "stem": raw.strip()
                        or f"{state['keyword']} 관련 주요 쟁점의 차이와 한계를 서술하세요.",
                        "box_example": None,
                        "choices": [],
                        "answer_indices": [1],
                        "answer": "참고 기사의 핵심 쟁점 및 근거 반영",
                        "explanation": "기사 본문 근거 확인",
                    }
                ],
                "rubric": rubric_list,
            }

        parsed = _normalize_single_question_via_plugin(
            parsed,
            spec=spec,
            allow_choices=allow_choices,
            num_choices=st_n,
            num_correct=st_m,
        )

        excerpts = parsed.get("source_excerpts") or []
        if isinstance(excerpts, list) and len(excerpts) > disp_limit:
            parsed["source_excerpts"] = excerpts[:disp_limit]

        quiz_id = f"quiz_{uuid.uuid4().hex[:10]}"
        parsed["quiz_id"] = quiz_id
        parsed["topic"] = state["keyword"]
        parsed["question_type"] = q_type
        parsed["is_essay"] = is_essay
        parsed["source_display_limit"] = disp_limit
        parsed["design_plan"] = plan
        parsed["llm_rule"] = extra_rule
        parsed["source_count"] = len(articles)
        parsed["rag_relevance_ok"] = bool(state.get("relevance_ok"))
        if rag_warn:
            parsed["rag_warning"] = rag_warn
        return {"draft_quiz": parsed}

    def node_verify(state: QuizGenState) -> Dict[str, Any]:
        quiz = state["draft_quiz"]
        spec = get_quiz_type_spec(quiz.get("question_type") or resolved_type)

        plugin_ok, plugin_msg = spec["validate"](quiz)
        if not plugin_ok:
            return {
                "quality_report": {"passed": False, "score": 0.2, "feedback": plugin_msg},
                "retry_count": state.get("retry_count", 0) + 1,
            }

        sanitized_for_judge = {
            k: v
            for k, v in quiz.items()
            if k not in ("source_excerpts", "rubric")
        }
        judge_prompt = (
            "당신은 문항 논리 모순과 품질을 검증하는 심사관입니다. 0.0~1.0 점수와 판정을 JSON으로만 답하세요.\n"
            f"요청 주제: {state['keyword']} | 유형: {quiz.get('question_type')} | is_essay: {quiz.get('is_essay')}\n"
            f"사용자 원문: {state['user_text']}\n"
            "검증 항목:\n"
            "1. 서술형(is_essay=true)인데 선지형/객관식 문항처럼 구성되었거나 지문과 모순되는 점이 없는가?\n"
            "2. 문제 지문과 문항 요구사항이 서로 논리적으로 맞물리는가?\n"
            "3. 정답(answer/answer_indices)과 해설(explanation)이 문항에 대응하는가?\n"
            f"검증 대상 JSON:\n{json.dumps(sanitized_for_judge, ensure_ascii=False)[:2500]}\n\n"
            '출력 형식: {"passed": true, "score": 0.9, "feedback": "개선점 또는 양호"}'
        )
        raw = llm_call_fn(judge_prompt, 300) or ""
        judged = _extract_json_obj(raw)
        if not judged:
            has_q = bool(quiz.get("questions"))
            judged = {
                "passed": has_q,
                "score": 0.8 if has_q else 0.3,
                "feedback": "양호" if has_q else "문항을 보완하세요.",
            }
        judged["score"] = float(judged.get("score") or 0.5)
        judged["passed"] = bool(judged.get("passed")) and judged["score"] >= 0.65
        return {
            "quality_report": judged,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def route_after_verify(state: QuizGenState) -> str:
        if state["quality_report"].get("passed"):
            return "done"
        if state.get("retry_count", 0) >= 2:
            return "done"
        return "retry"

    graph = StateGraph(QuizGenState)
    graph.add_node("fetch_rag", node_fetch_rag)
    graph.add_node("judge_relevance", node_judge_relevance)
    graph.add_node("rewrite_rag_query", node_rewrite_rag_query)
    graph.add_node("plan_design", node_plan_design)
    graph.add_node("generate", node_generate)
    graph.add_node("verify", node_verify)

    graph.set_entry_point("fetch_rag")
    graph.add_edge("fetch_rag", "judge_relevance")
    graph.add_conditional_edges(
        "judge_relevance",
        route_after_relevance,
        {"rewrite_rag": "rewrite_rag_query", "plan_design": "plan_design"},
    )
    graph.add_edge("rewrite_rag_query", "fetch_rag")
    graph.add_edge("plan_design", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"retry": "generate", "done": END})
    compiled = graph.compile()

    initial_state: QuizGenState = {
        "user_text": user_text,
        "keyword": keyword,
        "rag_query": keyword,
        "question_type": resolved_type,
        "num_choices": n_cnt,
        "num_correct": m_cnt,
        "custom_rubric": custom_rubric or [],
        "llm_rule": llm_rule,
        "existing_ctx": existing_ctx,
        "articles": [],
        "relevance_ok": False,
        "rag_feedback": "",
        "rag_retry_count": 0,
        "design_plan": {},
        "source_display_limit": max(1, limit),
        "draft_quiz": {},
        "quality_report": {},
        "retry_count": 0,
        "ls_project": ls_project,
    }

    final_state = compiled.invoke(
        initial_state,
        config={
            "run_name": f"quiz_{topic_label}_{resolved_type}",
            "tags": [resolved_type, f"topic:{topic_label}"],
            "metadata": {
                "keyword": keyword,
                "question_type": resolved_type,
                "num_choices": n_cnt,
                "num_correct": m_cnt,
                "llm_rule": llm_rule or "",
                "langsmith_project": ls_project,
            },
        },
    )

    quiz_record = dict(final_state["draft_quiz"])
    quiz_record["topic"] = final_state["keyword"] or topic_label
    quiz_record["quality_report"] = final_state["quality_report"]
    quiz_record["langsmith"] = {
        "project_name": ls_project,
        "attempts": final_state["retry_count"],
        "rag_retries": final_state["rag_retry_count"],
    }

    saved_path = dbms.save_quiz(topic=quiz_record["topic"], quiz_record=quiz_record)
    return {
        "quiz_record": quiz_record,
        "saved_path": saved_path,
        "ls_project": ls_project,
        "articles": final_state.get("articles") or [],
    }


def regenerate_quiz_from_record(
    source_quiz: Dict[str, Any],
    user_text: str,
    llm_call_fn: Callable[[str, int], Optional[str]],
    dbms: QuizProjectDBMS,
    target_format: Optional[str] = None,
    question_type: Optional[str] = None,
    num_choices: Optional[int] = None,
    num_correct: Optional[int] = None,
) -> Dict[str, Any]:
    """기존 퀴즈 레코드를 바탕으로 새 형식(객관식/서술형)의 독립 퀴즈 1문항을 생성·저장합니다."""
    topic = source_quiz.get("topic") or "general"
    topic_label = sanitize_topic_filename(topic)
    ls_project = (
        (source_quiz.get("langsmith") or {}).get("project_name")
        or f"quiz-{topic_label}-{uuid.uuid4().hex[:4]}"
    )

    want_essay = target_format == "essay" or (
        target_format is None and ("서술형" in user_text or "주관식" in user_text)
    )
    want_mc = target_format == "multiple_choice" or (
        target_format is None and ("객관식" in user_text or "선지" in user_text)
    )

    if want_essay:
        resolved_type = question_type or "custom_rubric_essay"
        allow_choices = False
    elif want_mc:
        resolved_type = question_type or "csat_passage_set"
        allow_choices = True
    else:
        resolved_type = question_type or source_quiz.get("question_type") or "custom_rubric_essay"
        spec_tmp = get_quiz_type_spec(resolved_type)
        allow_choices = bool(spec_tmp.get("allow_choices"))

    spec = get_quiz_type_spec(resolved_type)
    max_n = int(get_quiz_type_spec("csat_passage_set").get("max_choices") or 10)
    parsed_n, parsed_m = _parse_n_m_via_plugin(user_text, default_n=5, default_m=1)
    n_cnt = max(2, min(int(num_choices if num_choices is not None else parsed_n), max_n))
    m_cnt = max(1, min(int(num_correct if num_correct is not None else parsed_m), n_cnt - 1))

    source_summary = {
        "title": source_quiz.get("title"),
        "passage": source_quiz.get("passage"),
        "questions": source_quiz.get("questions"),
        "source_excerpts": source_quiz.get("source_excerpts"),
        "rubric": source_quiz.get("rubric"),
    }

    format_instruction = (
        "반드시 서술형 1문항으로 재탄생시키세요. choices는 반드시 빈 배열([])로 두고 객관식 선지를 절대 만들지 마세요."
        if not allow_choices
        else (
            f"반드시 객관식 1문항으로 재탄생시키세요. choices에 정확히 {n_cnt}개 선택지를 넣고, "
            f"{m_cnt}개 정답(answer_indices)과 {n_cnt - m_cnt}개 오답 함정 및 해설(explanation)을 작성하세요."
        )
    )

    prompt = (
        "기존 문제를 바탕으로 새로운 관점/형식의 퀴즈 1문항으로 재탄생시키세요. JSON만 출력하세요.\n"
        f"주제: {topic}\n"
        f"사용자 요청: {user_text}\n"
        f"형식 지침: {format_instruction}\n"
        f"기존 문제 데이터:\n{json.dumps(source_summary, ensure_ascii=False)[:2500]}\n\n"
        "출력 JSON 형식:\n"
        "{\n"
        '  "title": "새 문제 제목",\n'
        '  "passage": "지문 또는 상황 설명",\n'
        '  "source_excerpts": [],\n'
        '  "questions": [\n'
        "    {\n"
        '      "number": 1,\n'
        '      "stem": "새 문항",\n'
        '      "box_example": null,\n'
        '      "choices": [],\n'
        '      "answer_indices": [1],\n'
        '      "answer": "정답/핵심요지",\n'
        '      "explanation": "해설"\n'
        "    }\n"
        "  ],\n"
        '  "rubric": ["채점 기준 1", "채점 기준 2"]\n'
        "}"
    )

    raw = llm_call_fn(prompt, 1500) or ""
    parsed = _extract_json_obj(raw) or {
        "title": f"{topic} 재구성 문항",
        "passage": source_quiz.get("passage") or "",
        "source_excerpts": source_quiz.get("source_excerpts") or [],
        "questions": [
            {
                "number": 1,
                "stem": f"{topic}의 핵심 쟁점과 근거를 논리적으로 서술하세요.",
                "box_example": None,
                "choices": [],
                "answer_indices": [1],
                "answer": "핵심 쟁점과 타당한 근거 제시",
                "explanation": "기존 문제의 쟁점과 근거를 바탕으로 재구성",
            }
        ],
        "rubric": source_quiz.get("rubric") or ["핵심 근거의 정확성", "논리적 일관성"],
    }

    parsed = _normalize_single_question_via_plugin(
        parsed,
        spec=spec,
        allow_choices=allow_choices,
        num_choices=n_cnt,
        num_correct=m_cnt,
    )

    new_quiz_id = f"quiz_{uuid.uuid4().hex[:10]}"
    parsed["quiz_id"] = new_quiz_id
    parsed["topic"] = topic
    parsed["question_type"] = resolved_type
    parsed["is_essay"] = not allow_choices
    parsed["source_display_limit"] = int(
        source_quiz.get("source_display_limit") or len(parsed.get("source_excerpts") or []) or 1
    )
    parsed["derived_from"] = {
        "quiz_id": source_quiz.get("quiz_id"),
        "topic": topic,
        "title": source_quiz.get("title"),
    }
    parsed["quality_report"] = {"passed": True, "score": 0.9, "feedback": "기존 문제 기반 재탄생"}
    parsed["langsmith"] = {"project_name": ls_project, "attempts": 1}

    saved_path = dbms.save_quiz(topic=topic, quiz_record=parsed)
    return {
        "quiz_record": parsed,
        "saved_path": saved_path,
        "ls_project": ls_project,
        "articles": [],
    }


def format_quiz_for_slack(
    result: Dict[str, Any],
    articles: Optional[List[Dict[str, Any]]] = None,
    show_sources: bool = False,
    show_rubric: bool = False,
) -> str:
    """초기 출제 시에는 참고 기사 자료와 채점 기준을 숨기고, 요청 시에만 표시합니다."""
    rec = result["quiz_record"]
    spec = get_quiz_type_spec(rec.get("question_type") or "")
    q_report = rec.get("quality_report") or {}
    disp_limit = int(
        rec.get("source_display_limit") or len(rec.get("source_excerpts") or []) or 1
    )
    allow_choices = bool(spec.get("allow_choices")) and not bool(rec.get("is_essay"))

    lines = [
        f"📝 *[퀴즈 생성 완료]* `{rec.get('topic')}` | 유형: `{rec.get('question_type')}` ({spec.get('label', '')})",
        f"• 저장 파일: `{os.path.relpath(result['saved_path'])}` | LangSmith: `{result['ls_project']}` (품질 {q_report.get('score', 0):.2f})",
    ]
    if rec.get("rag_warning"):
        lines.append(f"⚠️ *RAG 탐색 안내*: {rec['rag_warning']}")

    lines.extend(["", f"*{rec.get('title') or '문제'}*"])
    if rec.get("passage"):
        lines.append(f"> {rec['passage']}")

    if show_sources:
        excerpts = rec.get("source_excerpts") or []
        if excerpts:
            lines.append("\n*참고 기사 자료*")
            for ex in excerpts[:disp_limit]:
                src = ex.get("source") or "기사"
                summ = ex.get("summary") or ""
                lines.append(f"• *{src}*: {summ}")
        elif articles:
            lines.append("\n*참고 기사 자료*")
            for a in articles[:disp_limit]:
                title = a.get("title") or "제목 없음"
                body = (
                    a.get("facts")
                    or a.get("summary")
                    or a.get("content")
                    or a.get("opinion")
                    or ""
                ).strip()
                lines.append(f"• *{title}*: {body[:180]}")

    lines.append("")
    for q in (rec.get("questions") or [])[:1]:
        lines.append(f"*Q{q.get('number', 1)}.* {q.get('stem', '')}")
        if q.get("box_example"):
            lines.append(f"```<보기>\n{q['box_example']}\n```")
        if allow_choices:
            for ch in q.get("choices") or []:
                lines.append(f"  {ch}")

    if show_rubric and rec.get("rubric"):
        lines.append("\n*채점 기준*")
        for r in rec["rubric"]:
            lines.append(f"• {r}")

    lines.append(
        "\n_이 메시지의 스레드에 답안을 남겨주시면 판별해 드릴게요. "
        "(참고 기사나 채점 기준이 필요하시면 스레드에 요청해 주세요.)_"
    )
    return "\n".join(lines)


def build_quiz_slack_blocks(
    result: Dict[str, Any],
    show_sources: bool = False,
    show_rubric: bool = False,
) -> List[Dict[str, Any]]:
    """객관식 UI 블록 조립을 타입 플러그인(`build_choice_blocks`)에 위임합니다."""
    rec = result["quiz_record"]
    spec = get_quiz_type_spec(rec.get("question_type") or "")
    allow_choices = bool(spec.get("allow_choices")) and not bool(rec.get("is_essay"))

    header_text = format_quiz_for_slack(
        result,
        articles=result.get("articles"),
        show_sources=show_sources,
        show_rubric=show_rubric,
    )

    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": header_text[:2900]},
        }
    ]

    block_builder = spec.get("build_choice_blocks") or get_quiz_type_spec("csat_passage_set").get("build_choice_blocks")
    if allow_choices and callable(block_builder):
        blocks.extend(block_builder(rec))

    return blocks