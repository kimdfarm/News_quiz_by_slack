import os
import uuid
import random
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langsmith import traceable

QUIZ_TYPES = [
    "multi_perspective",     # 1. 다각도 언론 비교·논증 서술형
    "csat_passage_set",      # 2. 수능/비문학형 지문 + 2~3문항 세트
    "custom_rubric_essay",   # 3. 커스텀 루브릭 기반 코칭 서술형
    "fact_check_trap",       # 4. 팩트체크 & 함정 선지 판별형
]


class QuizGenState(TypedDict):
    user_request: str
    keyword: Optional[str]
    question_type: Optional[str]
    custom_rubric: Optional[List[str]]
    rag_documents: List[Dict[str, Any]]
    draft_quiz: Dict[str, Any]
    quality_report: Dict[str, Any]
    retry_count: int
    langsmith_project: str


def node_resolve_keyword_and_rag(state: QuizGenState) -> Dict[str, Any]:
    """키워드가 없으면 RAG 저장소에서 랜덤 키워드를 추출하고 관련 기사를 모음"""
    kw = state.get("keyword")
    if not kw:
        # TODO: 실제 RAG/언론 DB의 최근 토픽 목록에서 랜덤 선택 함수 연결
        kw = fetch_random_keyword_from_rag()

    q_type = state.get("question_type") or random.choice(QUIZ_TYPES)
    docs = search_rag_encoder_docs(keyword=kw, top_k=4)  # 기존 RAG 인코더 검색기 연결
    return {
        "keyword": kw,
        "question_type": q_type,
        "rag_documents": docs,
    }


def node_generate_quiz(state: QuizGenState) -> Dict[str, Any]:
    """선택된 4가지 유형 템플릿에 맞춰 구조화된 문제(지문/문항/정답/루브릭) 생성"""
    quiz_id = f"quiz_{uuid.uuid4().hex[:10]}"
    # TODO: LLM 호출로 유형별 JSON 생성 (이전 검증 실패 피드백이 있으면 함께 전달)
    feedback = state.get("quality_report", {}).get("feedback", "")
    generated = call_llm_quiz_template(
        keyword=state["keyword"],
        question_type=state["question_type"],
        docs=state["rag_documents"],
        custom_rubric=state.get("custom_rubric"),
        improvement_hint=feedback,
    )
    generated["quiz_id"] = quiz_id
    generated["topic"] = state["keyword"]
    generated["question_type"] = state["question_type"]
    generated["source_articles"] = [
        {"title": d.get("title"), "press": d.get("press"), "date": d.get("date")}
        for d in state["rag_documents"]
    ]
    return {"draft_quiz": generated}


def node_verify_quality(state: QuizGenState) -> Dict[str, Any]:
    """출제 품질 검증: 원문 왜곡 여부, 정답 유일성, 루브릭 명확성 채점"""
    report = call_llm_quality_judge(
        quiz=state["draft_quiz"],
        docs=state["rag_documents"],
    )
    # report 예: {"passed": True, "score": 0.92, "feedback": "근거와 선지가 명확함"}
    return {
        "quality_report": report,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def route_quality(state: QuizGenState) -> str:
    if state["quality_report"].get("passed", False):
        return "accept"
    if state.get("retry_count", 0) >= 2:
        return "accept"  # 최대 2회 재생성 후 종료
    return "retry"


def build_quiz_graph():
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
    keyword: Optional[str] = None,
    question_type: Optional[str] = None,
    custom_rubric: Optional[List[str]] = None,
    custom_ls_project: Optional[str] = None,
) -> Dict[str, Any]:
    """문제마다 동적 LangSmith 프로젝트명으로 실행하고 최종 JSON 레코드를 반환"""
    topic_label = sanitize_topic_filename(keyword or "random_topic")
    ls_project = custom_ls_project or f"quiz-{topic_label}-{uuid.uuid4().hex[:4]}"

    # 실행 시점에만 동적으로 LangSmith 프로젝트 지정
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = ls_project

    graph = build_quiz_graph()
    final_state = graph.invoke(
        {
            "user_request": user_request,
            "keyword": keyword,
            "question_type": question_type,
            "custom_rubric": custom_rubric,
            "rag_documents": [],
            "draft_quiz": {},
            "quality_report": {},
            "retry_count": 0,
            "langsmith_project": ls_project,
        },
        config={
            "run_name": f"create_{topic_label}",
            "tags": [question_type or "auto", f"topic:{topic_label}"],
            "metadata": {"keyword": keyword, "ls_project": ls_project},
        },
    )

    quiz_record = final_state["draft_quiz"]
    quiz_record["quality_report"] = final_state["quality_report"]
    quiz_record["langsmith"] = {
        "project_name": ls_project,
        "retry_count": final_state["retry_count"],
    }
    return quiz_record