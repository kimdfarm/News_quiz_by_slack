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


class QuizGenState(TypedDict):
    user_request: str
    keyword: Optional[str]
    question_type: Optional[str]
    num_choices: int
    num_correct: int
    custom_rubric: List[str]
    rag_documents: List[Dict[str, Any]]
    draft_quiz: Dict[str, Any]
    quality_report: Dict[str, Any]
    retry_count: int
    langsmith_project: str


def _extract_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def _resolve_type(
    user_request: str,
    requested_type: Optional[str],
    llm_call_fn: Callable[[str, int], Optional[str]],
) -> str:
    available = load_all_quiz_types()
    if requested_type and requested_type in available:
        if "서술형" in user_request and available[requested_type].get("allow_choices"):
            return "multi_perspective" if "multi_perspective" in available else requested_type
        return requested_type

    if "서술형" in user_request:
        return "multi_perspective" if "multi_perspective" in available else random.choice(list(available.keys()))

    catalog = [
        {"name": k, "label": v.get("label", k), "description": v.get("description", "")}
        for k, v in available.items()
    ]
    prompt = (
        "사용자 요청에 맞는 문제 유형(name) 1개를 고르세요. JSON만 출력하세요.\n"
        '출력 형식: {"question_type": "유형name"}\n\n'
        f"요청: {user_request}\n"
        f"유형 목록: {json.dumps(catalog, ensure_ascii=False)}"
    )
    picked = (_extract_json(llm_call_fn(prompt, 150)) or {}).get("question_type")
    return picked if picked in available else ("csat_passage_set" if "csat_passage_set" in available else random.choice(list(available.keys())))


def build_quiz_graph(
    fetch_articles_fn: Callable[[Optional[str], int], List[Dict[str, Any]]],
    llm_call_fn: Callable[[str, int], Optional[str]],
    resolved_type: str,
    top_k: int = 4,
):
    def node_resolve_keyword_and_rag(state: QuizGenState) -> Dict[str, Any]:
        kw = (state.get("keyword") or "").strip() or None
        docs = fetch_articles_fn(kw, max(top_k * 3, 12))
        if not docs and kw:
            docs = fetch_articles_fn(None, max(top_k * 3, 12))

        if not kw and docs:
            words = [
                w
                for a in docs[:8]
                for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", str(a.get("title") or ""))
                if w not in ("뉴스", "기사", "논란", "발표", "전망", "시장")
            ]
            kw = random.choice(words) if words else "시사_이슈"

        return {
            "keyword": kw or "시사_이슈",
            "question_type": state.get("question_type") or resolved_type,
            "rag_documents": (docs or [])[:top_k],
        }

    def node_generate_quiz(state: QuizGenState) -> Dict[str, Any]:
        q_type = state.get("question_type") or resolved_type
        spec = get_quiz_type_spec(q_type)
        is_essay = bool(spec.get("is_essay")) or (not bool(spec.get("allow_choices"))) or ("서술형" in state["user_request"])
        allow_choices = bool(spec.get("allow_choices")) and not is_essay

        rubric = state.get("custom_rubric") or [
            "핵심 쟁점과 근거를 정확히 제시했는가",
            "기사 사실과 모순되는 주장이 없는가",
            "논리적 일관성이 있는가",
        ]
        feedback = ((state.get("quality_report") or {}).get("feedback")) or ""
        n_cnt = int(state.get("num_choices") or 5)
        m_cnt = int(state.get("num_correct") or 1)

        ctx_lines = []
        for i, d in enumerate(state.get("rag_documents") or [], 1):
            ctx_lines.append(
                f"[{i}] ({d.get('source') or d.get('press') or '언론'}) {d.get('title') or ''}\n"
                f"사실: {d.get('facts') or d.get('summary') or d.get('content') or ''}\n"
                f"관점: {d.get('opinion') or ''}"
            )
        ctx = "\n".join(ctx_lines) or "(참고 기사 없음)"

        guide = spec["build_guide"](rubric) if callable(spec.get("build_guide")) else ""
        format_rule = (
            "서술형 1문항입니다. choices는 반드시 빈 배열([])로 두세요."
            if not allow_choices
            else f"객관식 1문항입니다. choices에 {n_cnt}개 선택지, answer_indices에 {m_cnt}개 정답 번호(1부터 시작)를 넣으세요."
        )

        prompt = (
            f"주제: {state['keyword']} | 유형: {q_type}\n"
            f"사용자 요청: {state['user_request']}\n"
            f"출제 지침: {guide}\n"
            f"형식 규칙: {format_rule}\n"
            + (f"보완 피드백: {feedback}\n" if feedback else "")
            + f"참고 자료:\n{ctx}\n\n"
            'JSON만 출력: {"title": "", "passage": "", "questions": [{"number": 1, "stem": "", "choices": [], "answer_indices": [1], "answer": "", "explanation": ""}], "rubric": []}'
        )
        parsed = _extract_json(llm_call_fn(prompt, 1500)) or {
            "title": f"{state['keyword']} 문제",
            "passage": ctx[:600],
            "questions": [
                {
                    "number": 1,
                    "stem": f"{state['keyword']}의 핵심 쟁점을 분석하세요.",
                    "choices": [],
                    "answer_indices": [1],
                    "answer": "참고 자료 기반 핵심 요지",
                    "explanation": "참고 자료 근거 확인",
                }
            ],
            "rubric": rubric,
        }

        if allow_choices and callable(spec.get("normalize_record")):
            parsed = spec["normalize_record"](parsed, num_choices=n_cnt, num_correct=m_cnt)
        elif not allow_choices:
            q0 = dict(((parsed.get("questions") or [{}])[0]) or {})
            q0["number"] = 1
            q0["choices"] = []
            parsed["questions"] = [q0]

        parsed["quiz_id"] = f"quiz_{uuid.uuid4().hex[:10]}"
        parsed["topic"] = state["keyword"]
        parsed["question_type"] = q_type
        parsed["is_essay"] = not allow_choices
        parsed["source_articles"] = [
            {
                "title": d.get("title"),
                "source": d.get("source") or d.get("press"),
                "date": d.get("published_at") or d.get("date"),
            }
            for d in (state.get("rag_documents") or [])
        ]
        return {"draft_quiz": parsed}

    def node_verify_quality(state: QuizGenState) -> Dict[str, Any]:
        quiz = state.get("draft_quiz") or {}
        spec = get_quiz_type_spec(quiz.get("question_type") or resolved_type)
        validator = spec.get("validate")
        if callable(validator):
            ok, msg = validator(quiz)
            if not ok:
                return {
                    "quality_report": {"passed": False, "score": 0.2, "feedback": msg},
                    "retry_count": int(state.get("retry_count") or 0) + 1,
                }

        has_q = bool(quiz.get("questions"))
        return {
            "quality_report": {
                "passed": has_q,
                "score": 0.9 if has_q else 0.2,
                "feedback": "양호" if has_q else "문항 누락",
            },
            "retry_count": int(state.get("retry_count") or 0) + 1,
        }

    def route_quality(state: QuizGenState) -> str:
        if (state.get("quality_report") or {}).get("passed", False):
            return "accept"
        if int(state.get("retry_count") or 0) >= 2:
            return "accept"
        return "retry"

    workflow = StateGraph(QuizGenState)
    workflow.add_node("resolve_rag", node_resolve_keyword_and_rag)
    workflow.add_node("generate_quiz", node_generate_quiz)
    workflow.add_node("verify_quality", node_verify_quality)

    workflow.set_entry_point("resolve_rag")
    workflow.add_edge("resolve_rag", "generate_quiz")
    workflow.add_edge("generate_quiz", "verify_quality")
    workflow.add_conditional_edges(
        "verify_quality",
        route_quality,
        {"retry": "generate_quiz", "accept": END},
    )
    return workflow.compile()


def run_quiz_creation(
    user_request: str,
    fetch_articles_fn: Callable[[Optional[str], int], List[Dict[str, Any]]],
    llm_call_fn: Callable[[str, int], Optional[str]],
    dbms: Optional[QuizProjectDBMS] = None,
    keyword: Optional[str] = None,
    question_type: Optional[str] = None,
    custom_rubric: Optional[List[str]] = None,
    custom_ls_project: Optional[str] = None,
    top_k: int = 4,
) -> Dict[str, Any]:
    resolved_type = _resolve_type(user_request, question_type, llm_call_fn)
    csat_spec = get_quiz_type_spec("csat_passage_set")
    parse_nm = csat_spec.get("parse_n_m")
    n_cnt, m_cnt = parse_nm(user_request) if callable(parse_nm) else (5, 1)

    topic_label = sanitize_topic_filename(keyword or f"random_{resolved_type}")
    ls_project = custom_ls_project or f"quiz-{topic_label}-{uuid.uuid4().hex[:4]}"

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = ls_project

    compiled = build_quiz_graph(
        fetch_articles_fn=fetch_articles_fn,
        llm_call_fn=llm_call_fn,
        resolved_type=resolved_type,
        top_k=top_k,
    )
    final_state = compiled.invoke(
        {
            "user_request": user_request,
            "keyword": keyword,
            "question_type": resolved_type,
            "num_choices": n_cnt,
            "num_correct": m_cnt,
            "custom_rubric": custom_rubric or [],
            "rag_documents": [],
            "draft_quiz": {},
            "quality_report": {},
            "retry_count": 0,
            "langsmith_project": ls_project,
        },
        config={
            "run_name": f"create_{topic_label}",
            "tags": [resolved_type, f"topic:{topic_label}"],
            "metadata": {"keyword": keyword, "ls_project": ls_project},
        },
    )

    quiz_record = dict(final_state["draft_quiz"])
    quiz_record["topic"] = final_state.get("keyword") or topic_label
    quiz_record["quality_report"] = final_state["quality_report"]
    quiz_record["langsmith"] = {
        "project_name": ls_project,
        "retry_count": final_state["retry_count"],
    }

    saved_path = dbms.save_quiz(topic=quiz_record["topic"], quiz_record=quiz_record) if dbms else None
    return {
        "quiz_record": quiz_record,
        "saved_path": saved_path,
        "ls_project": ls_project,
        "articles": final_state.get("rag_documents") or [],
    }