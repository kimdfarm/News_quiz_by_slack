import json
import os
import random
import re
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from QUIZ.types import get_quiz_type_names, get_quiz_type_spec, load_all_quiz_types

load_all_quiz_types()
QUIZ_TYPES = get_quiz_type_names()


def sanitize_topic_filename(topic: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]+', "", topic or "").strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:60] or "random_quiz"


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


class QuizProjectDBMS:
    """PROJECT/<문제주제>.json 파일을 DBMS처럼 다루는 인코더/디코더 저장소"""

    def __init__(self, base_dir: str = "PROJECT"):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.index_path = os.path.join(self.base_dir, "_index.json")
        if not os.path.exists(self.index_path):
            self._atomic_write(self.index_path, {"threads": {}, "quizzes": {}})

    def _atomic_write(self, path: str, payload: Dict[str, Any]) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _read_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def topic_file_path(self, topic: str) -> str:
        return os.path.join(self.base_dir, f"{sanitize_topic_filename(topic)}.json")

    def list_topics(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for fn in sorted(os.listdir(self.base_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            fp = os.path.join(self.base_dir, fn)
            doc = self._read_json(fp)
            out.append({
                "file": fn,
                "topic": doc.get("topic"),
                "count": len(doc.get("records") or []),
                "updated_at": doc.get("updated_at"),
            })
        return out

    def save_quiz(
        self,
        topic: str,
        quiz_record: Dict[str, Any],
        thread_ts: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> str:
        file_path = self.topic_file_path(topic)
        now = datetime.now().isoformat()

        if os.path.exists(file_path):
            doc = self._read_json(file_path)
        else:
            doc = {
                "schema_version": "1.0",
                "topic": topic,
                "created_at": now,
                "updated_at": now,
                "records": [],
            }

        quiz_id = quiz_record["quiz_id"]
        quiz_record["saved_at"] = now
        if thread_ts:
            quiz_record["slack"] = {"channel_id": channel_id, "thread_ts": thread_ts}

        idx = next(
            (i for i, r in enumerate(doc["records"]) if r.get("quiz_id") == quiz_id),
            None,
        )
        if idx is not None:
            doc["records"][idx] = quiz_record
        else:
            doc["records"].append(quiz_record)

        doc["updated_at"] = now
        self._atomic_write(file_path, doc)

        index_data = self._read_json(self.index_path)
        index_data["quizzes"][quiz_id] = {
            "topic": topic,
            "file": os.path.basename(file_path),
            "question_type": quiz_record.get("question_type"),
            "langsmith_project": (quiz_record.get("langsmith") or {}).get("project_name"),
        }
        if thread_ts:
            index_data["threads"][str(thread_ts)] = {
                "quiz_id": quiz_id,
                "topic": topic,
                "file": os.path.basename(file_path),
                "channel_id": channel_id,
            }
        self._atomic_write(self.index_path, index_data)
        return file_path

    def get_by_thread(self, thread_ts: str) -> Optional[Dict[str, Any]]:
        index_data = self._read_json(self.index_path)
        meta = (index_data.get("threads") or {}).get(str(thread_ts))
        if not meta:
            return None
        items = self.select_quizzes(topic=meta.get("topic"), quiz_id=meta.get("quiz_id"))
        return items[0] if items else None

    def select_quizzes(
        self,
        topic: Optional[str] = None,
        quiz_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        paths = (
            [self.topic_file_path(topic)]
            if topic
            else [
                os.path.join(self.base_dir, fn)
                for fn in sorted(os.listdir(self.base_dir))
                if fn.endswith(".json") and not fn.startswith("_")
            ]
        )
        matched: List[Dict[str, Any]] = []
        for fp in paths:
            if not os.path.exists(fp):
                continue
            doc = self._read_json(fp)
            for rec in doc.get("records") or []:
                if quiz_id and rec.get("quiz_id") != quiz_id:
                    continue
                matched.append(rec)
        return matched

    def delete_quiz(self, quiz_id: str) -> bool:
        index_data = self._read_json(self.index_path)
        meta = (index_data.get("quizzes") or {}).get(quiz_id)
        if not meta:
            return False
        fp = os.path.join(self.base_dir, meta["file"])
        if os.path.exists(fp):
            doc = self._read_json(fp)
            doc["records"] = [r for r in (doc.get("records") or []) if r.get("quiz_id") != quiz_id]
            doc["updated_at"] = datetime.now().isoformat()
            self._atomic_write(fp, doc)
        index_data["quizzes"].pop(quiz_id, None)
        index_data["threads"] = {
            ts: info for ts, info in (index_data.get("threads") or {}).items() if info.get("quiz_id") != quiz_id
        }
        self._atomic_write(self.index_path, index_data)
        return True

    def export_bundle(
        self,
        export_path: str,
        topics: Optional[List[str]] = None,
        template_style: str = "csat_2col",
    ) -> str:
        items: List[Dict[str, Any]] = []
        target_topics = topics or [t["topic"] for t in self.list_topics() if t.get("topic")]
        for t in target_topics:
            items.extend(self.select_quizzes(topic=t))
        bundle = {
            "workbook_meta": {
                "style": template_style,
                "exported_at": datetime.now().isoformat(),
                "total_items": len(items),
            },
            "items": items,
        }
        self._atomic_write(export_path, bundle)
        return export_path

    def import_bundle(self, external_json_path: str) -> int:
        raw = self._read_json(external_json_path)
        records = raw.get("records") or raw.get("items") or []
        fallback_topic = raw.get("topic") or os.path.splitext(os.path.basename(external_json_path))[0]
        count = 0
        for rec in records:
            if not isinstance(rec, dict) or not rec.get("quiz_id"):
                continue
            topic = rec.get("topic") or fallback_topic
            slack_info = rec.get("slack") or {}
            self.save_quiz(
                topic=topic,
                quiz_record=rec,
                thread_ts=slack_info.get("thread_ts"),
                channel_id=slack_info.get("channel_id"),
            )
            count += 1
        return count

    def get_existing_summaries(self, limit: int = 20) -> List[str]:
        summaries: List[str] = []
        for rec in self.select_quizzes():
            q_type = rec.get("question_type", "")
            topic = rec.get("topic", "")
            title = rec.get("title", "")
            summaries.append(f"[{topic} | {q_type}] {title}")
            if len(summaries) >= limit:
                break
        return summaries


class QuizGenState(TypedDict):
    user_text: str
    keyword: Optional[str]
    rag_query: Optional[str]
    question_type: Optional[str]
    custom_rubric: List[str]
    llm_rule: Optional[str]
    existing_ctx: str
    articles: List[Dict[str, Any]]
    relevance_ok: bool
    rag_feedback: str
    rag_retry_count: int
    draft_quiz: Dict[str, Any]
    quality_report: Dict[str, Any]
    retry_count: int
    ls_project: str


def _articles_to_context(articles: List[Dict[str, Any]]) -> str:
    if not articles:
        return "(저장된 기사 데이터 중 요청 주제와 직접 일치하는 기사가 부족합니다.)"
    lines = []
    for i, a in enumerate(articles[:6], 1):
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
        # 사용자가 명시적으로 "서술형"을 요구했는데 객관식 유형이 들어온 경우 보정
        if "서술형" in user_text and available[requested_type].get("allow_choices"):
            return "multi_perspective" if "multi_perspective" in available else requested_type
        return requested_type

    if "서술형" in user_text:
        return "multi_perspective" if "multi_perspective" in available else random.choice(list(available.keys()))

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
) -> Dict[str, Any]:
    resolved_type = _infer_type_with_llm(user_text, question_type, llm_call_fn)
    type_spec = get_quiz_type_spec(resolved_type)

    topic_label = sanitize_topic_filename(keyword or f"random_{resolved_type}")
    ls_project = f"quiz-{topic_label}-{uuid.uuid4().hex[:4]}"

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = ls_project

    existing_summaries = dbms.get_existing_summaries(limit=20)
    existing_ctx = "\n".join(f"- {s}" for s in existing_summaries)

    def node_fetch_rag(state: QuizGenState) -> Dict[str, Any]:
        query = state.get("rag_query") or state.get("keyword")
        # 후보군을 넉넉히 조회해서 LangGraph가 정밀 필터링하도록 전달
        articles = fetch_articles_fn(query, max(limit * 3, 12))
        if not articles and query:
            articles = fetch_articles_fn(None, max(limit * 3, 12))
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

        # 특정 키워드/주제 지정이 없으면 바로 통과
        if not target_topic:
            picked = articles[:limit]
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
                "articles": filtered[:limit],
                "relevance_ok": True,
                "rag_feedback": "",
            }

        return {
            "articles": [],
            "relevance_ok": False,
            "rag_retry_count": retry_cnt + 1,
            "rag_feedback": str(parsed.get("reason") or f"'{target_topic}' 주제와 직접 관련된 기사를 찾지 못했습니다."),
        }

    def node_rewrite_rag_query(state: QuizGenState) -> Dict[str, Any]:
        prev_q = state.get("rag_query") or state.get("keyword") or ""
        reason = state.get("rag_feedback") or ""
        prompt = (
            "이전 RAG 검색에서 요청 주제와 맞는 기사를 찾지 못했습니다.\n"
            "DB에서 관련 기사를 정밀하게 찾기 위한 대체 핵심 검색어(1단어 또는 짧은 키워드) 1개를 제안하세요.\n"
            "예: 'AI' -> '인공지능', 'LLM', '생성형', '반도체'\n"
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
            return "generate"
        if int(state.get("rag_retry_count") or 0) >= 2:
            return "generate"
        return "rewrite_rag"

    def node_generate(state: QuizGenState) -> Dict[str, Any]:
        articles = state.get("articles") or []
        ctx = _articles_to_context(articles)
        q_type = state["question_type"] or resolved_type
        spec = get_quiz_type_spec(q_type)
        allow_choices = bool(spec.get("allow_choices")) and ("서술형" not in state["user_text"])

        prev_feedback = (state.get("quality_report") or {}).get("feedback") or ""
        rag_warn = state.get("rag_feedback") or ""
        extra_rule = state.get("llm_rule") or ""
        rubric_list = state.get("custom_rubric") or [
            "핵심 쟁점 또는 근거를 정확히 제시했는가",
            "기사 사실과 모순되는 주장이 없는가",
        ]
        type_guide = spec["build_guide"](rubric_list)
        format_rule = (
            "객관식 문항이므로 choices 배열에 4~5개 선택지를 반드시 포함하세요."
            if allow_choices
            else "서술형/주관식 문항이므로 choices는 반드시 빈 배열([])로 출력하고 객관식 선지를 절대 생성하지 마세요."
        )

        prompt = (
            f"주제 키워드: {state['keyword']}\n"
            f"문제 유형: {q_type} ({spec.get('label', q_type)})\n"
            f"사용자 요청: {state['user_text']}\n"
            + (f"최종 판별/출제 규칙: {extra_rule}\n" if extra_rule else "")
            + (f"이미 PROJECT에 생성된 문제 목록(가능하면 겹치지 않게 출제):\n{state.get('existing_ctx', '')}\n" if state.get("existing_ctx") else "")
            + f"참고 기사:\n{ctx}\n\n"
            f"출제 지침: {type_guide}\n"
            f"형식 강제 규칙: {format_rule}\n"
            "중요: 참고 기사에 없는 내용을 억지로 기사 보도인 것처럼 지어내지 마세요.\n"
            + (f"참고 알림: {rag_warn}\n" if rag_warn else "")
            + (f"이전 품질 검증 피드백(반드시 반영해 개선): {prev_feedback}\n" if prev_feedback else "")
            + "\n반드시 아래 JSON 형식으로만 출력하세요:\n"
            "{\n"
            '  "title": "문제 세트 제목",\n'
            '  "passage": "지문 또는 상황 설명(구체적 기사 요약·발췌 포함)",\n'
            '  "source_excerpts": [\n'
            '    {"source": "언론사/제목", "summary": "사용자가 답을 도출할 수 있는 핵심 사실·관점 요약"}\n'
            "  ],\n"
            '  "questions": [\n'
            '    {"number": 1, "stem": "문항", "box_example": null, "choices": [], "answer": "정답/핵심요지", "explanation": "해설"}\n'
            "  ],\n"
            '  "rubric": ["채점 기준 1", "채점 기준 2"]\n'
            "}"
        )

        raw = llm_call_fn(prompt, 1600) or ""
        parsed = _extract_json_obj(raw)
        if not parsed:
            parsed = {
                "title": f"{state['keyword']} 퀴즈",
                "passage": ctx[:800],
                "questions": [
                    {
                        "number": 1,
                        "stem": raw.strip() or f"{state['keyword']} 핵심 내용을 서술하세요.",
                        "box_example": None,
                        "choices": [],
                        "answer": "참고 기사의 핵심 쟁점 반영",
                        "explanation": "기사 본문 근거 확인",
                    }
                ],
                "rubric": rubric_list,
            }

        # 서술형일 경우 모델이 실수로 넣은 choices 원천 제거
        if not allow_choices:
            for q in parsed.get("questions") or []:
                if isinstance(q, dict):
                    q["choices"] = []

        quiz_id = f"quiz_{uuid.uuid4().hex[:10]}"
        parsed["quiz_id"] = quiz_id
        parsed["topic"] = state["keyword"]
        parsed["question_type"] = q_type
        parsed["llm_rule"] = extra_rule
        parsed["source_count"] = len(articles)
        parsed["rag_relevance_ok"] = bool(state.get("relevance_ok"))
        if rag_warn:
            parsed["rag_warning"] = rag_warn
        return {"draft_quiz": parsed}

    def node_verify(state: QuizGenState) -> Dict[str, Any]:
        quiz = state["draft_quiz"]
        spec = get_quiz_type_spec(quiz.get("question_type") or resolved_type)

        # 1단계: 유형 플러그인의 구조적 검증 규칙 실행
        plugin_ok, plugin_msg = spec["validate"](quiz)
        if not plugin_ok:
            return {
                "quality_report": {"passed": False, "score": 0.2, "feedback": plugin_msg},
                "retry_count": state.get("retry_count", 0) + 1,
            }

        # 2단계: LLM 품질/주제 정합성 평가
        judge_prompt = (
            "다음 퀴즈 초안의 품질을 0.0~1.0 점수로 평가하고 JSON으로만 답하세요.\n"
            f"요청 주제: {state['keyword']} | 요청 유형: {quiz.get('question_type')} | 사용자 원문: {state['user_text']}\n"
            "평가 기준: 요청 주제 일치 여부, 문항 형식 일치(서술형/객관식), 정답/해설/루브릭 존재 여부.\n"
            f"퀴즈 JSON:\n{json.dumps(quiz, ensure_ascii=False)[:2500]}\n\n"
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
    graph.add_node("generate", node_generate)
    graph.add_node("verify", node_verify)

    graph.set_entry_point("fetch_rag")
    graph.add_edge("fetch_rag", "judge_relevance")
    graph.add_conditional_edges(
        "judge_relevance",
        route_after_relevance,
        {"rewrite_rag": "rewrite_rag_query", "generate": "generate"},
    )
    graph.add_edge("rewrite_rag_query", "fetch_rag")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"retry": "generate", "done": END})
    compiled = graph.compile()

    initial_state: QuizGenState = {
        "user_text": user_text,
        "keyword": keyword,
        "rag_query": keyword,
        "question_type": resolved_type,
        "custom_rubric": custom_rubric or [],
        "llm_rule": llm_rule,
        "existing_ctx": existing_ctx,
        "articles": [],
        "relevance_ok": False,
        "rag_feedback": "",
        "rag_retry_count": 0,
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


def format_quiz_for_slack(result: Dict[str, Any], articles: Optional[List[Dict[str, Any]]] = None) -> str:
    rec = result["quiz_record"]
    spec = get_quiz_type_spec(rec.get("question_type") or "")
    q_report = rec.get("quality_report") or {}

    lines = [
        f"📝 *[퀴즈 생성 완료]* `{rec.get('topic')}` | 유형: `{rec.get('question_type')}` ({spec.get('label', '')})",
        f"• 저장 파일: `{os.path.relpath(result['saved_path'])}` | LangSmith: `{result['ls_project']}` (품질 {q_report.get('score', 0):.2f})",
    ]
    if rec.get("rag_warning"):
        lines.append(f"⚠️ *RAG 탐색 안내*: {rec['rag_warning']}")

    lines.extend(["", f"*{rec.get('title') or '문제'}*"])
    if rec.get("passage"):
        lines.append(f"> {rec['passage']}")

    excerpts = rec.get("source_excerpts") or []
    if excerpts:
        lines.append("\n*참고 기사 자료*")
        for ex in excerpts[:4]:
            src = ex.get("source") or "기사"
            summ = ex.get("summary") or ""
            lines.append(f"• *{src}*: {summ}")
    elif articles:
        lines.append("\n*참고 기사 자료*")
        for a in articles[:4]:
            title = a.get("title") or "제목 없음"
            body = (a.get("facts") or a.get("summary") or a.get("content") or a.get("opinion") or "").strip()
            lines.append(f"• *{title}*: {body[:180]}")

    lines.append("")
    for q in rec.get("questions") or []:
        lines.append(f"*Q{q.get('number', '')}.* {q.get('stem', '')}")
        if q.get("box_example"):
            lines.append(f"```<보기>\n{q['box_example']}\n```")
        if spec.get("allow_choices"):
            for ch in q.get("choices") or []:
                lines.append(f"  {ch}")

    if rec.get("rubric"):
        lines.append("\n*채점 기준*")
        for r in rec["rubric"]:
            lines.append(f"• {r}")

    lines.append("\n_위 참고 기사 자료를 바탕으로 이 메시지의 스레드에 답안을 남겨주시면 참·거짓/루브릭으로 판별해 드릴게요._")
    return "\n".join(lines)