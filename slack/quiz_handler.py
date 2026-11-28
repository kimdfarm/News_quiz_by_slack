from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

from langchain_core.messages import HumanMessage

from QUIZ.hybrid_rag_store import (
    DEFAULT_COOKIE_RAG_FAISS_DIR,
    FAISSHybridRAGStore,
    PostgresHybridRAGStore,
    _article_doc_key,
    decide_index_sync_timing_with_llm,
    sync_rag_index_standalone,
)
from QUIZ.quiz_engine import (
    build_quiz_slack_blocks,
    format_quiz_for_slack,
    generate_and_save_quiz_with_graph,
    regenerate_quiz_from_record,
)
from QUIZ.quiz_json_dbms import QuizProjectDBMS
from QUIZ.quiz_semantic_classifier import is_rebuttal_semantic
from QUIZ.types import get_quiz_type_spec
from slack.llm_router import (
    clean_str_arg,
    extract_json,
    invoke_llm_with_fallback,
    resolve_positive_int_with_llm,
    resolve_quiz_keyword_with_llm,
    unwrap_tool_args,
)
from storage_backend import DEFAULT_SQLITE_PATH, get_active_storage_backend

_ROOT_DIR = Path(__file__).resolve().parent.parent
_PROJECT_DIR = str(_ROOT_DIR / "PROJECT")
quiz_dbms = QuizProjectDBMS(base_dir=_PROJECT_DIR)


def get_quiz_rag_faiss_dir() -> str:
  """FAISS 벡터 인덱스가 저장될 로컬 디렉터리 경로(기본: cookie/faiss_quiz_rag)를 반환합니다."""
  env_path = (
      os.getenv("QUIZ_RAG_FAISS_DIR") or os.getenv("QUIZ_RAG_SQLITE_PATH") or ""
  ).strip()
  if env_path:
    p = Path(env_path)
    if p.suffix:
      p = p.parent / "faiss_quiz_rag"
    return str(p if p.is_absolute() else (_ROOT_DIR / p))
  return str(_ROOT_DIR / "cookie" / "faiss_quiz_rag")


get_quiz_rag_sqlite_path = get_quiz_rag_faiss_dir


def _parse_n_m(user_text: str) -> Tuple[int, int]:
  csat_spec = get_quiz_type_spec("csat_passage_set")
  parser = csat_spec.get("parse_n_m")
  if callable(parser):
    return parser(user_text, default_n=5, default_m=1)
  return 5, 1


def _llm_for_quiz_graph(prompt: str, max_tokens: int) -> Optional[str]:
  prefer_local = (
      os.getenv("QUIZ_LLM_PROVIDER") or ""
  ).strip().lower() == "ollama"
  return invoke_llm_with_fallback(
      messages=[HumanMessage(content=prompt)],
      temperature=0.3,
      max_tokens=max_tokens,
      prefer_local=prefer_local,
  )


def read_storage_mode() -> str:
  """storage_mode_state.json의 storage_mode 값을 최우선으로 읽습니다."""
  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  candidates = [
      os.path.join(base_dir, "storage_mode_state.json"),
      os.path.join(
          os.path.dirname(os.path.abspath(__file__)), "storage_mode_state.json"
      ),
      os.path.join(os.getcwd(), "storage_mode_state.json"),
  ]
  for path in candidates:
    if os.path.isfile(path):
      try:
        with open(path, "r", encoding="utf-8") as f:
          data = json.load(f)
        mode = str((data or {}).get("storage_mode") or "").strip().lower()
        if mode:
          return mode
      except Exception as e:
        print(f"[storage_mode_state.json 읽기 실패 ({path})]: {e}")

  try:
    return str(get_active_storage_backend() or "sqlite").strip().lower()
  except Exception:
    return "sqlite"


def fetch_articles_for_quiz(
    keyword: Optional[str],
    limit: int,
    mcp_tools: Any,
    user_text: str = "",
) -> List[Dict[str, Any]]:
  """
    storage_mode_state.json에 따라:
    - Neon(PostgreSQL): pgvector + pgvectorscale + FTS
    - SQLite/로컬: 원본 뉴스는 news_store.sqlite에서 읽고, 벡터 인덱스는 FAISS(cookie/faiss_quiz_rag)에 저장·검색
    """
  mode = read_storage_mode()
  top_k = max(limit * 2, 8)
  query_str = user_text or (keyword or "")

  if mode in ("postgres", "postgresql", "neon") and os.getenv("DATABASE_URL"):
    try:
      from database import get_db_connection

      store = PostgresHybridRAGStore()
      conn = get_db_connection()
      cur = conn.cursor()
      try:
        if store.ensure_schema(cur):
          indexed_cnt, pending = store.get_status_and_pending(
              cur, scan_limit=100
          )
          decision = decide_index_sync_timing_with_llm(
              keyword=keyword,
              user_text=query_str,
              storage_mode="postgres(pgvector+pgvectorscale)",
              indexed_count=indexed_cnt,
              pending_articles=pending,
              llm_call_fn=_llm_for_quiz_graph,
          )
          if decision["action"] == "incremental_now" and pending:
            store.upsert_articles(cur, pending)
            conn.commit()
          elif decision["action"] == "defer_sync":
            print(f"[RAG 별도 동기화 권장 - Postgres]: {decision['reason']}")

          hits = store.hybrid_search(
              cur=cur,
              query_text=query_str,
              keyword=keyword,
              top_k=top_k,
          )
          if hits:
            return hits[:limit]
      finally:
        cur.close()
        conn.close()
    except Exception as e:
      print(f"[Quiz Postgres 하이브리드 RAG 실패, 로컬 FAISS로 폴백]: {e}")

  raw_res = mcp_tools.get_recent_news_for_quiz(
      category="전체",
      limit=120,
      sqlite_path=DEFAULT_SQLITE_PATH,
  )
  bundle = json.loads(raw_res) if raw_res else {}
  recent_articles = list(bundle.get("articles") or [])

  if keyword and keyword.strip():
    cat_res = mcp_tools.get_recent_news_for_quiz(
        category=keyword,
        limit=60,
        sqlite_path=DEFAULT_SQLITE_PATH,
    )
    cat_bundle = json.loads(cat_res) if cat_res else {}
    seen_keys = {_article_doc_key(a) for a in recent_articles}
    for art in cat_bundle.get("articles") or []:
      dk = _article_doc_key(art)
      if dk not in seen_keys:
        recent_articles.append(art)
        seen_keys.add(dk)

  faiss_dir = get_quiz_rag_faiss_dir()
  store = FAISSHybridRAGStore(index_dir=faiss_dir)
  store.ensure_schema()
  existing_keys = store.get_indexed_keys()
  pending = [
      a for a in recent_articles if _article_doc_key(a) not in existing_keys
  ]

  idx_label = "faiss+kw" if store._faiss_available else "numpy+kw"
  if len(existing_keys) == 0 and pending:
    store.upsert_articles(pending)
    pending = []
  elif pending:
    decision = decide_index_sync_timing_with_llm(
        keyword=keyword,
        user_text=query_str,
        storage_mode=f"local({Path(faiss_dir).name}:{idx_label})",
        indexed_count=len(existing_keys),
        pending_articles=pending,
        llm_call_fn=_llm_for_quiz_graph,
    )
    if decision.get("action") == "incremental_now":
      store.upsert_articles(pending)
      pending = []
    elif decision.get("action") == "defer_sync":
      print(f"[RAG 별도 동기화 권장 - FAISS]: {decision.get('reason')}")

  hits = store.hybrid_search(
      query_text=query_str,
      keyword=keyword,
      top_k=top_k,
  )

  if not hits and pending:
    store.upsert_articles(pending)
    hits = store.hybrid_search(
        query_text=query_str,
        keyword=keyword,
        top_k=top_k,
    )

  return (hits or recent_articles)[:limit]


def _extract_choice_numbers(user_answer: str, max_choices: int) -> List[int]:
  csat_spec = get_quiz_type_spec("csat_passage_set")
  extractor = csat_spec.get("extract_choice_numbers")
  if callable(extractor):
    return extractor(user_answer, max_choices=max_choices)

  circle_nums = csat_spec.get("circle_nums") or [
      "①",
      "②",
      "③",
      "④",
      "⑤",
      "⑥",
      "⑦",
      "⑧",
      "⑨",
      "⑩",
  ]
  limit_n = max(1, min(int(max_choices or len(circle_nums)), len(circle_nums)))
  s = (user_answer or "").strip()
  found: List[int] = []
  for idx, sym in enumerate(circle_nums[:limit_n], 1):
    if sym in s and idx not in found:
      found.append(idx)
  for m in re.finditer(r"(?:^|[^\d])([1-9]|10)(?:\s*번|[^\d]|$)", s):
    num = int(m.group(1))
    if 1 <= num <= limit_n and num not in found:
      found.append(num)
  return sorted(found)


def grade_multiple_choice_deterministic(
    quiz_record: Dict[str, Any],
    selected_indices: Optional[List[int]],
    raw_user_text: str,
    user_id: str,
) -> str:
  spec = get_quiz_type_spec(quiz_record.get("question_type") or "")
  csat_spec = get_quiz_type_spec("csat_passage_set")
  grader = (
      spec.get("grade_mc")
      if callable(spec.get("grade_mc"))
      else csat_spec.get("grade_mc")
  )
  if callable(grader):
    return grader(
        quiz_record=quiz_record,
        selected_indices=selected_indices,
        raw_user_text=raw_user_text,
        user_id=user_id,
    )

  q = (quiz_record.get("questions") or [{}])[0]
  choices = q.get("choices") or []
  n_cnt = len(choices)
  raw_ans_indices = q.get("answer_indices")
  ans_indices = (
      sorted({int(x) for x in raw_ans_indices if 1 <= int(x) <= n_cnt})
      if isinstance(raw_ans_indices, list) and raw_ans_indices
      else ([1] if n_cnt >= 1 else [])
  )
  chosen_list = (
      sorted({int(x) for x in selected_indices if 1 <= int(x) <= n_cnt})
      if selected_indices is not None
      else _extract_choice_numbers(raw_user_text, max_choices=n_cnt)
  )
  is_correct = chosen_list == ans_indices
  chosen_labels = ", ".join(choices[i - 1] for i in chosen_list)
  ans_labels = ", ".join(choices[i - 1] for i in ans_indices)
  return (
      f"<@{user_id}> 님의 답: *{chosen_labels}* | "
      f"{'⭕ 정답입니다!' if is_correct else f'❌ 오답입니다. 정답: *{ans_labels}*'}"
  )


def _grade_thread_submission(
    quiz_record: Dict[str, Any], user_answer: str, user_id: str
) -> str:
  q = (quiz_record.get("questions") or [{}])[0]
  choices = q.get("choices") or []
  is_mc = bool(choices) and not bool(quiz_record.get("is_essay"))

  if is_mc:
    return grade_multiple_choice_deterministic(
        quiz_record=quiz_record,
        selected_indices=None,
        raw_user_text=user_answer,
        user_id=user_id,
    )

  ls_project = (quiz_record.get("langsmith") or {}).get("project_name")
  if ls_project:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = ls_project

  rubric = quiz_record.get("rubric") or []
  prompt = (
      "당신은 서술형 퀴즈 루브릭 채점관입니다. JSON으로만 답하세요.\n"
      "중요 규칙: is_true가 false이면 reason에 절대 '정답입니다', '일치합니다'"
      " 같은 표현을 쓰지 마세요.\n"
      f"문항: {q.get('stem')}\n"
      f"모범 핵심 요지: {q.get('answer')}\n"
      f"채점 기준: {json.dumps(rubric, ensure_ascii=False)}\n"
      f"제출자(<@{user_id}>) 답안: {user_answer}\n\n"
      '출력 형식: {"is_true": true, "score_text": "기준 충족/미충족",'
      ' "reason": "판별 이유", "missing": []}'
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
    try:
      parsed = (
          json.loads(m.group(0))
          if m
          else {"is_true": False, "reason": raw.strip()}
      )
    except Exception:
      parsed = {"is_true": False, "reason": raw.strip()}

  verdict_label = (
      "참 (기준 충족)" if parsed.get("is_true") else "거짓 (기준 미충족)"
  )
  lines = [
      (
          f"<@{user_id}> 답안 판별: *{verdict_label}* "
          f"({parsed.get('score_text', '')})"
      ),
      f"• 근거: {parsed.get('reason', '')}",
  ]
  if parsed.get("missing"):
    lines.append(
        f"• 보완할 점: {', '.join(str(x) for x in parsed['missing'])}"
    )
  return "\n".join(lines)


def _format_requested_quiz_extras(
    quiz_record: Dict[str, Any], req_text: str
) -> Optional[str]:
  low = req_text.lower()
  want_rubric = any(
      k in low
      for k in (
          "채점 기준",
          "채점기준",
          "루브릭",
          "기준 보여",
          "기준 알려",
          "기준도",
      )
  )
  want_sources = any(
      k in low
      for k in (
          "참고 기사",
          "참고자료",
          "참고 자료",
          "기사 보여",
          "기사 알려",
          "출처",
          "참고도",
      )
  )

  if not want_rubric and not want_sources:
    return None

  parts: List[str] = []
  if want_sources:
    disp_limit = int(
        quiz_record.get("source_display_limit")
        or len(quiz_record.get("source_excerpts") or [])
        or 1
    )
    excerpts = (quiz_record.get("source_excerpts") or [])[:disp_limit]
    parts.append("*참고 기사 자료*")
    if excerpts:
      for ex in excerpts:
        parts.append(
            f"• *{ex.get('source') or '기사'}*: {ex.get('summary') or ''}"
        )
    else:
      parts.append("• 저장된 기사 발췌가 없습니다.")

  if want_rubric:
    rubric = quiz_record.get("rubric") or []
    parts.append("*채점 기준*")
    if rubric:
      for r in rubric:
        parts.append(f"• {r}")
    else:
      parts.append("• 등록된 채점 기준이 없습니다.")

  return "\n".join(parts)


def _build_quiz_evidence_json(quiz_record: Dict[str, Any]) -> Dict[str, Any]:
  """문제 JSON 레코드에서 반박/재해석 판단에 필요한 근거만 구조화합니다."""
  q_list = quiz_record.get("questions") or []
  first_q = q_list[0] if q_list and isinstance(q_list[0], dict) else {}
  return {
      "quiz_id": quiz_record.get("quiz_id"),
      "topic": quiz_record.get("topic"),
      "question_type": quiz_record.get("question_type"),
      "is_essay": bool(quiz_record.get("is_essay")),
      "title": quiz_record.get("title"),
      "flow_outline": quiz_record.get("flow_outline") or "",
      "passage": quiz_record.get("passage") or "",
      "question": {
          "number": first_q.get("number", 1),
          "stem": first_q.get("stem") or "",
          "box_example": first_q.get("box_example"),
          "choices": first_q.get("choices") or [],
          "answer_indices": first_q.get("answer_indices") or [],
          "answer": first_q.get("answer") or "",
          "explanation": first_q.get("explanation") or "",
      },
      "rubric": quiz_record.get("rubric") or [],
      "source_excerpts": quiz_record.get("source_excerpts") or [],
  }


def _record_rebuttal_trace_in_quiz(
    quiz_record: Dict[str, Any],
    user_id: str,
    user_message: str,
    reply_text: str,
    ls_project: str,
    run_id: str,
) -> None:
  """문제 JSON 레코드에도 반박 검토 및 LangSmith 추적 정보를 남깁니다."""
  logs = list(quiz_record.get("rebuttal_logs") or [])
  logs.append({
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "user_id": user_id,
      "user_message": user_message[:1000],
      "assistant_reply": reply_text[:1500],
      "langsmith_project": ls_project,
      "langsmith_run_id": run_id,
  })
  quiz_record["rebuttal_logs"] = logs[-20:]
  ls_meta = dict(quiz_record.get("langsmith") or {})
  ls_meta["project_name"] = ls_project
  ls_meta["last_rebuttal_run_id"] = run_id
  ls_meta["rebuttal_count"] = len(quiz_record["rebuttal_logs"])
  quiz_record["langsmith"] = ls_meta


def _is_rebuttal_or_followup(text: str) -> bool:
  t = (text or "").strip()
  if not t:
    return False
  if re.fullmatch(r"[①-⑩]|(?:[1-9]|10)\s*번?", t):
    return False
  is_reb, _, _ = is_rebuttal_semantic(t, threshold=0.42)
  return is_reb


def _answer_quiz_rebuttal_with_context(
    quiz_record: Dict[str, Any],
    user_message: str,
    user_id: str,
) -> str:
  topic = str(quiz_record.get("topic") or "general")
  quiz_id = str(quiz_record.get("quiz_id") or "")
  ls_project = (
      str((quiz_record.get("langsmith") or {}).get("project_name") or "").strip()
      or f"quiz-rebuttal-{topic}"
  )
  run_id = str(uuid.uuid4())

  os.environ["LANGCHAIN_TRACING_V2"] = "true"
  os.environ["LANGCHAIN_PROJECT"] = ls_project

  evidence_json = _build_quiz_evidence_json(quiz_record)
  prompt = (
      "당신은 퀴즈 출제자이자 비문학 논증 검토관입니다.\n"
      "제공된 문제 JSON(지문, 문항, 선택지, 정답, 해설, 채점 기준, 기사 요약)만을"
      " 근거로 사용자의 반박이나 재질문을 논리적으로 분석하고 반박/재해석"
      " 응답을 작성하세요.\n\n"
      "판단 및 작성 규칙:\n"
      "1. [근거 대조] 사용자 주장이 지문/기사 요약/해설/채점 기준에 부합하는지,"
      " 아니면 지문에 없는 전제·확대해석·논점 일탈인지 먼저 구분하세요.\n"
      "2. [논리적 반박 또는 인정] 기존 정답·해설이 지문 근거상 타당하다면"
      " 반박의 허점을 짚어 논리적으로 방어하세요. 반대로 문항·해설·선지에"
      " 실제 모순이나 해석 여지가 있다면 억지로 우기지 말고 인정한 뒤"
      " 수정·보완 해석을 제시하세요.\n"
      "3. [형식] 감정적 표현 없이 핵심 판정 → 지문/JSON 근거 → 반박/재해석"
      " 결론 순으로 명료하게 작성하세요.\n\n"
      f"[문제 근거 JSON]\n{json.dumps(evidence_json, ensure_ascii=False)[:3800]}\n\n"
      f"[사용자(<@{user_id}>) 반박/질문]\n{user_message}"
  )

  invoke_kwargs: Dict[str, Any] = {
      "messages": [HumanMessage(content=prompt)],
      "temperature": 0.2,
      "max_tokens": 1100,
      "prefer_local": False,
  }
  try:
    reply = invoke_llm_with_fallback(
        **invoke_kwargs,
        config={
            "run_name": "quiz_rebuttal_review",
            "run_id": run_id,
            "tags": ["quiz_rebuttal", f"topic:{topic}", f"quiz:{quiz_id}"],
            "metadata": {
                "quiz_id": quiz_id,
                "topic": topic,
                "user_id": user_id,
                "langsmith_project": ls_project,
                "rebuttal_text": user_message[:500],
            },
        },
    )
  except TypeError:
    reply = invoke_llm_with_fallback(**invoke_kwargs)

  final_reply = (
      reply
      or "제공된 지문과 문제 근거를 대조해 보았으나 추가 반박/해석 결과를"
      " 생성하지 못했습니다."
  ).strip()

  _record_rebuttal_trace_in_quiz(
      quiz_record=quiz_record,
      user_id=user_id,
      user_message=user_message,
      reply_text=final_reply,
      ls_project=ls_project,
      run_id=run_id,
  )

  return f"{final_reply}\n\n_🔎 LangSmith 추적: `{ls_project}` (`{run_id[:8]}`)_"


def try_handle_quiz_thread_reply(
    thread_ts: str, answer_text: str, user_id: str, say: Any
) -> bool:
  if not thread_ts or not answer_text:
    return False
  matched_quiz = quiz_dbms.get_by_thread(str(thread_ts))
  if not matched_quiz:
    return False

  extras_msg = _format_requested_quiz_extras(matched_quiz, answer_text)
  if extras_msg:
    say(text=extras_msg, thread_ts=thread_ts)
    return True

  if _is_rebuttal_or_followup(answer_text):
    rebuttal_msg = _answer_quiz_rebuttal_with_context(
        quiz_record=matched_quiz,
        user_message=answer_text,
        user_id=user_id or "unknown",
    )
    say(text=rebuttal_msg, thread_ts=thread_ts)
    return True

  verdict_msg = _grade_thread_submission(
      quiz_record=matched_quiz,
      user_answer=answer_text,
      user_id=user_id or "unknown",
  )
  say(text=verdict_msg, thread_ts=thread_ts)
  return True


def _detect_regeneration_request(
    user_text: str, candidate_kw: Optional[str]
) -> Optional[Dict[str, Any]]:
  topics = quiz_dbms.list_topics()
  if not topics:
    return None

  topic_names = [str(t.get("topic") or "") for t in topics if t.get("topic")]
  prompt = (
      "사용자 요청이 기존 PROJECT 문제집/프로젝트의 문제를 바탕으로"
      " '새롭게/다시/재탄생/변형' 만들어 달라는 요청인지 판별하세요.\n"
      "JSON만 출력하세요.\n"
      '출력 형식: {"is_regen": true, "target_topic": "프로젝트명 또는 null",'
      ' "target_format": "essay | multiple_choice | null"}\n\n'
      f"현재 PROJECT 주제 목록: {json.dumps(topic_names, ensure_ascii=False)}\n"
      f"후보 키워드: {candidate_kw!r}\n"
      f"사용자 원문: {user_text!r}"
  )
  parsed = extract_json(
      invoke_llm_with_fallback(
          [HumanMessage(content=prompt)],
          temperature=0.0,
          max_tokens=150,
          prefer_local=True,
      )
      or ""
  )
  if not parsed or not parsed.get("is_regen"):
    return None

  raw_topic = str(parsed.get("target_topic") or candidate_kw or "").strip()
  matched_topic = None
  if raw_topic:
    for t_name in topic_names:
      if (
          t_name.lower() == raw_topic.lower()
          or raw_topic.lower() in t_name.lower()
          or t_name.lower() in raw_topic.lower()
      ):
        matched_topic = t_name
        break
    if not matched_topic:
      matched_topic = raw_topic

  records = (
      quiz_dbms.select_quizzes(topic=matched_topic)
      if matched_topic
      else quiz_dbms.select_quizzes()
  )
  if not records:
    return None

  fmt = str(parsed.get("target_format") or "").strip().lower()
  if fmt not in ("essay", "multiple_choice"):
    fmt = None

  return {
      "source_quiz": records[-1],
      "target_format": fmt,
  }


def _post_and_bind_thread(
    result: Dict[str, Any], event: Dict[str, Any], say: Any
) -> None:
  rec = result.get("quiz_record") or {}
  if result.get("aborted") or rec.get("aborted_no_articles"):
    warn_msg = (
        rec.get("rag_warning")
        or "수집한 RSS는 문제를 만들기 부족합니다. 다른 카테고리 혹은 뉴스"
        " 데이터를 더 가져오시오."
    )
    say(text=f"⚠️ {warn_msg}")
    return

  fallback_text = format_quiz_for_slack(
      result,
      articles=result.get("articles"),
      show_sources=False,
      show_rubric=False,
  )
  blocks = build_quiz_slack_blocks(result, show_sources=False, show_rubric=False)
  posted = say(text=fallback_text, blocks=blocks)
  posted_data = getattr(posted, "data", posted) if posted is not None else {}
  thread_ts = None
  if isinstance(posted_data, dict):
    thread_ts = posted_data.get("ts") or (posted_data.get("message") or {}).get(
        "ts"
    )
  if thread_ts:
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
  candidate_kw = clean_str_arg(merged.get("keyword")) or clean_str_arg(
      merged.get("category")
  )
  q_type = clean_str_arg(merged.get("question_type"))
  num_choices, num_correct = _parse_n_m(user_text)

  regen_info = _detect_regeneration_request(user_text, candidate_kw)
  if regen_info:
    say(
        text=(
            "♻️ 기존 프로젝트 문제를 바탕으로 *독립된 새"
            f" 문제*({num_choices}지선다·{num_correct}정답)로 재탄생시키는"
            " 중이에요..."
        )
    )
    regen_result = regenerate_quiz_from_record(
        source_quiz=regen_info["source_quiz"],
        user_text=user_text,
        llm_call_fn=_llm_for_quiz_graph,
        dbms=quiz_dbms,
        target_format=regen_info["target_format"],
        question_type=q_type,
        num_choices=num_choices,
        num_correct=num_correct,
    )
    _post_and_bind_thread(regen_result, event=event, say=say)
    return

  raw_kw = resolve_quiz_keyword_with_llm(candidate_kw, user_text=user_text)
  llm_rule = clean_str_arg(merged.get("llm_rule"))
  raw_rubric = merged.get("custom_rubric")
  rubric_list = (
      [str(x) for x in raw_rubric] if isinstance(raw_rubric, list) else None
  )
  limit = (
      resolve_positive_int_with_llm(
          merged.get("limit"),
          user_text=user_text,
          field_name="limit",
          default=4,
      )
      or 4
  )

  say(
      text=(
          "🧠 하이브리드 RAG + *LangGraph + LangSmith*로 1문항"
          f" 퀴즈({num_choices}지선다·{num_correct}정답)를 생성해 `PROJECT/`에"
          " 저장 중이에요..."
      )
  )
  result = generate_and_save_quiz_with_graph(
      user_text=user_text,
      fetch_articles_fn=lambda kw, lim: fetch_articles_for_quiz(
          kw,
          lim,
          mcp_tools=mcp_tools,
          user_text=user_text,
      ),
      llm_call_fn=_llm_for_quiz_graph,
      dbms=quiz_dbms,
      keyword=raw_kw,
      question_type=q_type,
      custom_rubric=rubric_list,
      llm_rule=llm_rule,
      limit=limit,
      num_choices=num_choices,
      num_correct=num_correct,
  )
  _post_and_bind_thread(result, event=event, say=say)


def handle_quiz_db_manage(
    event: Dict[str, Any],
    chosen: Dict[str, Any],
    say: Any,
    client: Any,
    mcp_tools: Any = None,
) -> None:
  merged = unwrap_tool_args(chosen)
  action = (clean_str_arg(merged.get("action")) or "list").lower()
  topic = clean_str_arg(merged.get("topic"))
  quiz_id = clean_str_arg(merged.get("quiz_id"))
  style = clean_str_arg(merged.get("template_style")) or "csat_2col"

  faiss_dir = get_quiz_rag_faiss_dir()

  if action in ("sync_rag", "sync_index", "reindex"):
    mode = read_storage_mode()
    candidates: List[Dict[str, Any]] = []
    if mode not in ("postgres", "postgresql", "neon") and mcp_tools is not None:
      raw_res = mcp_tools.get_recent_news_for_quiz(
          category="전체",
          limit=300,
          sqlite_path=DEFAULT_SQLITE_PATH,
      )
      bundle = json.loads(raw_res) if raw_res else {}
      candidates = list(bundle.get("articles") or [])
    sync_res = sync_rag_index_standalone(
        storage_mode=mode,
        index_dir=faiss_dir,
        candidate_articles=candidates,
    )
    say(
        text=(
            f"🔄 RAG 별도 동기화 완료 (`{sync_res['backend']}`, 신규 색인"
            f" `{sync_res['synced_count']}`건)"
        )
    )
    return

  if action == "list":
    topics = quiz_dbms.list_topics()
    if not topics:
      say(text="📂 `PROJECT/` 폴더에 저장된 문제 파일이 아직 없어요.")
      return
    lines = [
        f"• `{t['file']}` (주제: *{t['topic']}*, {t['count']}문항)"
        for t in topics
    ]
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
    status_msg = "완료" if quiz_dbms.delete_quiz(quiz_id) else "대상 없음"
    say(text=f"🗑️ `{quiz_id}` 삭제 {status_msg}.")
    return

  items = quiz_dbms.select_quizzes(topic=topic, quiz_id=quiz_id)
  say(text=f"🔎 조회 결과 `{len(items)}`건 (`topic={topic or '전체'}`)")


def handle_quiz_ui_submit(body: Dict[str, Any], client: Any) -> None:
  user_id = (body.get("user") or {}).get("id") or "unknown"
  channel_id = (body.get("channel") or {}).get("id")
  message = body.get("message") or {}
  thread_ts = message.get("thread_ts") or message.get("ts")

  actions = body.get("actions") or []
  quiz_id = actions[0].get("value") if actions else None

  state_values = (body.get("state") or {}).get("values") or {}
  block_key = f"quiz_select_block_{quiz_id}" if quiz_id else ""
  action_obj = state_values.get(block_key, {}).get("quiz_choice_select") or {}

  selected_values: List[str] = []
  if isinstance(action_obj.get("selected_options"), list):
    for opt in action_obj["selected_options"]:
      if isinstance(opt, dict) and opt.get("value"):
        selected_values.append(str(opt["value"]))
  elif isinstance(action_obj.get("selected_option"), dict) and action_obj[
      "selected_option"
  ].get("value"):
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
        text=(
            f"<@{user_id}> 먼저 위 선택지를 고른 뒤 *답안 제출하기*를 눌러주세요."
        ),
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


def handle_quiz_essay_modal_open(body: Dict[str, Any], client: Any) -> None:
  trigger_id = body.get("trigger_id")
  actions = body.get("actions") or []
  quiz_id = (actions[0].get("value") if actions else "") or ""

  channel_id = (body.get("channel") or {}).get("id") or ""
  message = body.get("message") or {}
  thread_ts = message.get("thread_ts") or message.get("ts") or ""

  matched_quiz = quiz_dbms.get_by_quiz_id(quiz_id) if quiz_id else None
  if not matched_quiz and thread_ts:
    matched_quiz = quiz_dbms.get_by_thread(str(thread_ts))

  q_stem = (
      ((matched_quiz or {}).get("questions") or [{}])[0]
  ).get("stem") or "서술형 답안을 작성해 주세요."
  slack_meta = (matched_quiz or {}).get("slack") or {}
  target_channel = channel_id or slack_meta.get("channel_id") or ""
  target_thread = thread_ts or slack_meta.get("thread_ts") or ""

  if not trigger_id:
    return

  client.views_open(
      trigger_id=trigger_id,
      view={
          "type": "modal",
          "callback_id": "quiz_essay_modal_submit",
          "private_metadata": json.dumps(
              {
                  "quiz_id": quiz_id,
                  "channel_id": target_channel,
                  "thread_ts": str(target_thread),
              },
              ensure_ascii=False,
          ),
          "title": {"type": "plain_text", "text": "서술형 답안 제출", "emoji": True},
          "submit": {"type": "plain_text", "text": "제출하기", "emoji": True},
          "close": {"type": "plain_text", "text": "취소", "emoji": True},
          "blocks": [
              {
                  "type": "section",
                  "text": {"type": "mrkdwn", "text": f"*문항:*\n{q_stem[:2800]}"},
              },
              {
                  "type": "input",
                  "block_id": "essay_answer_block",
                  "label": {
                      "type": "plain_text",
                      "text": "답안 작성",
                      "emoji": True,
                  },
                  "element": {
                      "type": "plain_text_input",
                      "action_id": "essay_answer_input",
                      "multiline": True,
                      "placeholder": {
                          "type": "plain_text",
                          "text": "핵심 쟁점과 근거를 바탕으로 답안을 입력하세요.",
                      },
                  },
              },
          ],
      },
  )


def handle_quiz_essay_modal_submit(body: Dict[str, Any], client: Any) -> None:
  user_id = (body.get("user") or {}).get("id") or "unknown"
  view = body.get("view") or {}
  meta_raw = view.get("private_metadata") or "{}"
  try:
    meta = json.loads(meta_raw)
  except Exception:
    meta = {}

  quiz_id = meta.get("quiz_id") or ""
  channel_id = meta.get("channel_id") or ""
  thread_ts = meta.get("thread_ts") or ""

  vals = view.get("state", {}).get("values", {})
  answer_text = (
      (
          ((vals.get("essay_answer_block") or {}).get("essay_answer_input") or {})
          .get("value")
      )
      or ""
  ).strip()

  matched_quiz = quiz_dbms.get_by_quiz_id(quiz_id) if quiz_id else None
  if not matched_quiz and thread_ts:
    matched_quiz = quiz_dbms.get_by_thread(str(thread_ts))

  if not matched_quiz or not channel_id or not answer_text:
    return

  verdict_msg = _grade_thread_submission(
      quiz_record=matched_quiz,
      user_answer=answer_text,
      user_id=user_id,
  )
  post_kwargs: Dict[str, Any] = {"channel": channel_id, "text": verdict_msg}
  if thread_ts:
    post_kwargs["thread_ts"] = str(thread_ts)
  client.chat_postMessage(**post_kwargs)