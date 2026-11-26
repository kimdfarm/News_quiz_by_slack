import json
import os
import random
import re
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

QUIZ_TYPES = [
    "multi_perspective",     # 1. 다각도 언론 비교·논증 서술형
    "csat_passage_set",      # 2. 수능/비문학형 지문 + 2~3문항 세트
    "custom_rubric_essay",   # 3. 커스텀 루브릭 기반 코칭 서술형
    "fact_check_trap",       # 4. 팩트체크 & 함정 선지 판별형
]


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
            index_data["threads"][thread_ts] = {
                "quiz_id": quiz_id,
                "topic": topic,
                "file": os.path.basename(file_path),
                "channel_id": channel_id,
            }
        self._atomic_write(self.index_path, index_data)
        return file_path

    def get_by_thread(self, thread_ts: str) -> Optional[Dict[str, Any]]:
        index_data = self._read_json(self.index_path)
        meta = (index_data.get("threads") or {}).get(thread_ts)
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


class QuizGenState(TypedDict):
    user_text: str
    keyword: Optional[str]
    question_type: Optional[str]
    custom_rubric: List[str]
    llm_rule: Optional[str]
    articles: List[Dict[str, Any]]
    draft_quiz: Dict[str, Any]
    quality_report: Dict[str, Any]
    retry_count: int
    ls_project: str


def _articles_to_context(articles: List[Dict[str, Any]]) -> str:
    if not articles:
        return "(저장된 기사 데이터가 부족하여 일반 시사 맥락을 참고합니다.)"
    lines = []
    for i, a in enumerate(articles[:6], 1):
        lines.append(
            f"[{i}] ({a.get('source') or a.get('press') or '언론'}) {a.get('title') or ''}\n"
            f"    사실: {a.get('facts') or a.get('summary') or a.get('content') or ''}\n"
            f"    관점({a.get('stance_type') or '중립'}): {a.get('opinion') or ''}"
        )
    return "\n".join(lines)


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
    # 라우터 LLM이 결정한 question_type/keyword/llm_rule을 그대로 존중하고, 없을 때만 랜덤으로 채움
    resolved_type = question_type if question_type in QUIZ_TYPES else random.choice(QUIZ_TYPES)
    topic_label = sanitize_topic_filename(keyword or f"random_{resolved_type}")
    ls_project = f"quiz-{topic_label}-{uuid.uuid4().hex[:4]}"

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = ls_project

    def node_fetch_rag(state: QuizGenState) -> Dict[str, Any]:
        kw = state.get("keyword")
        articles = fetch_articles_fn(kw, limit)
        if not articles and kw:
            articles = fetch_articles_fn(None, limit)
        if not kw:
            titles = [str(a.get("title") or "") for a in articles if a.get("title")]
            words = [
                w
                for t in titles
                for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", t)
                if w not in ("뉴스", "기사", "논란", "발표", "전망", "시장")
            ]
            kw = random.choice(words) if words else "시사_이슈"
        return {"keyword": kw, "articles": articles}

    def node_generate(state: QuizGenState) -> Dict[str, Any]:
        ctx = _articles_to_context(state["articles"])
        q_type = state["question_type"] or "csat_passage_set"
        prev_feedback = (state.get("quality_report") or {}).get("feedback") or ""
        extra_rule = state.get("llm_rule") or ""
        rubric_list = state.get("custom_rubric") or [
            "핵심 쟁점 또는 근거를 정확히 제시했는가",
            "기사 사실과 모순되는 주장이 없는가",
        ]

        type_guide = {
            "multi_perspective": (
                "다각도 언론 비교·논증 서술형 1문항을 만드세요. "
                "서로 다른 시각/언론의 핵심 차이를 묻고, 모범답안과 채점 기준(rubric)을 포함하세요."
            ),
            "csat_passage_set": (
                "수능/비문학형 지문 1개와 2~3개 문항 세트를 만드세요. "
                "<보기> 적용 문항을 최소 1개 포함하고, 각 문항의 선택지, 정답, 해설을 작성하세요."
            ),
            "custom_rubric_essay": (
                f"커스텀 루브릭 코칭 서술형 1문항을 만드세요. 적용할 채점 기준: {rubric_list}. "
                "답변 시 챙겨야 할 체크리스트와 힌트 단계도 함께 작성하세요."
            ),
            "fact_check_trap": (
                "기사 원문의 수치/주체/인과관계를 교묘하게 비튼 진술(함정)을 포함한 팩트체크 판별 문항을 만드세요. "
                "틀린 부분, 실제 사실 근거, 수정 예시를 함께 작성하세요."
            ),
        }.get(q_type, "")

        prompt = (
            f"주제 키워드: {state['keyword']}\n"
            f"문제 유형: {q_type}\n"
            f"사용자 요청: {state['user_text']}\n"
            + (f"최종 판별/출제 규칙: {extra_rule}\n" if extra_rule else "")
            + f"참고 기사:\n{ctx}\n\n"
            f"출제 지침: {type_guide}\n"
            + (f"이전 품질 검증 피드백(반드시 반영해 개선): {prev_feedback}\n" if prev_feedback else "")
            + "\n반드시 아래 JSON 형식으로만 출력하세요:\n"
            "{\n"
            '  "title": "문제 세트 제목",\n'
            '  "passage": "지문 또는 상황 설명",\n'
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

        quiz_id = f"quiz_{uuid.uuid4().hex[:10]}"
        parsed["quiz_id"] = quiz_id
        parsed["topic"] = state["keyword"]
        parsed["question_type"] = q_type
        parsed["llm_rule"] = extra_rule
        parsed["source_count"] = len(state["articles"])
        return {"draft_quiz": parsed}

    def node_verify(state: QuizGenState) -> Dict[str, Any]:
        quiz = state["draft_quiz"]
        judge_prompt = (
            "다음 퀴즈 초안의 품질을 0.0~1.0 점수로 평가하고 JSON으로만 답하세요.\n"
            "평가 기준: 문항 완성도, 정답/해설/루브릭 존재 여부, 질문의 명확성, 지정 규칙 반영 여부.\n"
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
    graph.add_node("generate", node_generate)
    graph.add_node("verify", node_verify)
    graph.set_entry_point("fetch_rag")
    graph.add_edge("fetch_rag", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"retry": "generate", "done": END})
    compiled = graph.compile()

    initial_state: QuizGenState = {
        "user_text": user_text,
        "keyword": keyword,
        "question_type": resolved_type,
        "custom_rubric": custom_rubric or [],
        "llm_rule": llm_rule,
        "articles": [],
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
    }

    saved_path = dbms.save_quiz(topic=quiz_record["topic"], quiz_record=quiz_record)
    return {
        "quiz_record": quiz_record,
        "saved_path": saved_path,
        "ls_project": ls_project,
    }


def format_quiz_for_slack(result: Dict[str, Any]) -> str:
    rec = result["quiz_record"]
    q_report = rec.get("quality_report") or {}
    lines = [
        f"📝 *[퀴즈 생성 완료]* `{rec.get('topic')}` | 유형: `{rec.get('question_type')}`",
        f"• 저장 파일: `{os.path.relpath(result['saved_path'])}` | LangSmith: `{result['ls_project']}` (품질 {q_report.get('score', 0):.2f})",
        "",
        f"*{rec.get('title') or '문제'}*",
    ]
    if rec.get("passage"):
        lines.append(f"> {rec['passage']}")
    lines.append("")

    for q in rec.get("questions") or []:
        lines.append(f"*Q{q.get('number', '')}.* {q.get('stem', '')}")
        if q.get("box_example"):
            lines.append(f"```<보기>\n{q['box_example']}\n```")
        for ch in q.get("choices") or []:
            lines.append(f"  {ch}")

    if rec.get("rubric"):
        lines.append("\n*채점 기준*")
        for r in rec["rubric"]:
            lines.append(f"• {r}")

    lines.append("\n_이 메시지의 스레드에 답안을 남겨주시면 참·거짓/루브릭으로 판별해 드릴게요._")
    return "\n".join(lines)