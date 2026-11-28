import json
import os
import random
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph
import numpy as np

from QUIZ.quiz_json_dbms import QuizProjectDBMS, sanitize_topic_filename
from QUIZ.quiz_semantic_classifier import _embed_normalized, classify_articles_flow
from QUIZ.types import get_quiz_type_names, get_quiz_type_spec, load_all_quiz_types

load_all_quiz_types()
QUIZ_TYPES = get_quiz_type_names()

DEFAULT_RUBRIC: List[str] = [
    "핵심 쟁점과 근거를 정확히 제시했는가",
    "기사 사실과 모순되는 주장이 없는가",
    "주장·비교·판단의 논리적 일관성이 있는가",
]

_MC_KEYWORDS = ("객관식", "선지", "지선다", "선택지", "사지선다", "오지선다", "육지선다")
_ESSAY_KEYWORDS = ("서술형", "주관식", "논술형")

_QUIZ_AXES: List[Dict[str, str]] = [
    {
        "name": "메커니즘·인과 추론",
        "guide": "현상이 발생한 구조적 원인과 연쇄적 파급효과를 중심으로 출제",
    },
    {
        "name": "한계·반례·리스크 비판",
        "guide": "주장이나 정책·기술 도입의 사각지대, 부작용, 반례 조건을 중심으로 출제",
    },
    {
        "name": "관점·이해관계 상충 비교",
        "guide": "서로 다른 입장이나 이해관계자의 전제 차이와 충돌 지점을 비교하도록 출제",
    },
    {
        "name": "사례 적용·대안 평가",
        "guide": "지문/기사의 원리나 쟁점을 구체적 상황이나 대응 방안에 적용·평가하도록 출제",
    },
]


def _wants_multiple_choice(user_text: str) -> bool:
  t = (user_text or "").strip()
  if any(k in t for k in _MC_KEYWORDS) and not any(
      k in t for k in _ESSAY_KEYWORDS
  ):
    return True
  if re.search(r"[2-9]|10\s*지선다", t):
    return True
  return False


def _wants_essay(user_text: str) -> bool:
  t = (user_text or "").strip()
  if any(k in t for k in _ESSAY_KEYWORDS) and not any(
      k in t for k in _MC_KEYWORDS
  ):
    return True
  return False


def _pick_mc_type_name(
    available: Dict[str, Dict[str, Any]], preferred: Optional[str] = None
) -> str:
  if (
      preferred
      and preferred in available
      and bool(available[preferred].get("allow_choices"))
  ):
    return preferred
  for cand in ("csat_passage_set", "multiple_choice", "objective"):
    if cand in available and bool(available[cand].get("allow_choices")):
      return cand
  for name, spec in available.items():
    if bool(spec.get("allow_choices")):
      return name
  return preferred or "csat_passage_set"


def _pick_essay_type_name(
    available: Dict[str, Dict[str, Any]], preferred: Optional[str] = None
) -> str:
  if (
      preferred
      and preferred in available
      and not bool(available[preferred].get("allow_choices"))
  ):
    return preferred
  for cand in ("multi_perspective", "custom_rubric_essay", "essay"):
    if cand in available and not bool(available[cand].get("allow_choices")):
      return cand
  return preferred or "custom_rubric_essay"


def _extract_json_obj(raw: Optional[str]) -> Optional[Dict[str, Any]]:
  if not raw:
    return None
  cleaned = re.sub(
      r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE
  ).strip()
  candidates = [cleaned]
  match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
  if match and match.group(0) != cleaned:
    candidates.append(match.group(0))

  for text in candidates:
    try:
      parsed = json.loads(text)
      if isinstance(parsed, dict):
        return parsed
    except Exception:
      continue
  return None


def _clamp_choice_counts(
    raw_choices: Optional[int],
    raw_correct: Optional[int],
    default_n: int,
    default_m: int,
    max_choices: int,
) -> Tuple[int, int]:
  max_n = max(3, int(max_choices or 10))
  try:
    n_val = int(raw_choices) if raw_choices is not None else default_n
  except (TypeError, ValueError):
    n_val = default_n
  n_cnt = max(2, min(n_val, max_n))

  try:
    m_val = int(raw_correct) if raw_correct is not None else default_m
  except (TypeError, ValueError):
    m_val = default_m
  m_cnt = max(1, min(m_val, n_cnt - 1))
  return n_cnt, m_cnt


def _parse_n_m_via_plugin(
    user_text: str,
    spec: Optional[Dict[str, Any]] = None,
    default_n: int = 5,
    default_m: int = 1,
) -> Tuple[int, int]:
  csat_spec = get_quiz_type_spec("csat_passage_set")
  active_spec = spec if (spec and callable(spec.get("parse_n_m"))) else csat_spec
  max_n = int(
      (spec or {}).get("max_choices") or csat_spec.get("max_choices") or 10
  )
  parser = active_spec.get("parse_n_m")
  if callable(parser):
    parsed_n, parsed_m = parser(
        user_text, default_n=default_n, default_m=default_m
    )
    return _clamp_choice_counts(parsed_n, parsed_m, default_n, default_m, max_n)
  return _clamp_choice_counts(default_n, default_m, default_n, default_m, max_n)


def _clean_stem_via_plugin(
    spec: Dict[str, Any],
    allow_choices: bool,
    raw_stem: str,
    passage: str = "",
    topic: str = "",
) -> str:
  """유형 플러그인에 stem 정제 훅이 있으면 위임하고, 없으면 공통 정제를 수행합니다."""
  stem = str(raw_stem or "").strip()
  if allow_choices and callable(spec.get("clean_stem")):
    return spec["clean_stem"](stem, passage=passage, topic=topic)
  if not stem:
    return (
        f"{topic or '제시된 주제'}의 핵심 쟁점과 한계를 논리적으로 서술하세요."
        if not allow_choices
        else f"{topic or '제시된 주제'} 관련 설명으로 가장 적절한 것은?"
    )
  if allow_choices:
    return re.split(r"\n\s*(?:[①-⑩]|(?:[1-9]|10)\.\s)", stem)[0].strip()
  return stem


def _normalize_single_question_via_plugin(
    parsed: Dict[str, Any],
    spec: Dict[str, Any],
    allow_choices: bool,
    num_choices: int = 5,
    num_correct: int = 1,
    topic: str = "",
) -> Dict[str, Any]:
  """1문항 고정 및 유형별 정규화 훅을 통해 레코드를 정규화합니다."""
  if allow_choices:
    if callable(spec.get("normalize_record")):
      norm = spec["normalize_record"](
          parsed,
          num_choices=num_choices,
          num_correct=num_correct,
      )
    else:
      csat_norm = get_quiz_type_spec("csat_passage_set").get("normalize_record")
      norm = (
          csat_norm(parsed, num_choices=num_choices, num_correct=num_correct)
          if callable(csat_norm)
          else dict(parsed)
      )

    qs = norm.get("questions") or []
    first_q = dict(qs[0]) if qs and isinstance(qs[0], dict) else {}
    choices = first_q.get("choices") or []
    stem_str = str(first_q.get("stem") or "").strip()
    if not choices or stem_str.startswith("{"):
      circle_nums = (
          spec.get("circle_nums")
          or get_quiz_type_spec("csat_passage_set").get("circle_nums")
          or ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
      )
      fallback_choices = [
          f"{circle_nums[i]} {topic or '제시 지문'} 핵심 내용에 부합하는"
          f" 설명({i + 1})"
          for i in range(num_choices)
      ]
      ans_idx_list = list(range(1, min(num_correct, num_choices) + 1))
      first_q.update({
          "number": 1,
          "stem": (
              f"제시된 지문과 참고 기사 내용을 바탕으로 {topic or '해당 주제'}"
              "에 대해 가장 적절한 것은?"
          ),
          "box_example": None,
          "choices": fallback_choices,
          "num_choices": num_choices,
          "num_correct": len(ans_idx_list),
          "answer_indices": ans_idx_list,
          "answer_index": ans_idx_list[0],
          "answer": ", ".join(fallback_choices[k - 1] for k in ans_idx_list),
          "explanation": (
              "제시된 지문과 기사의 사실 관계를 바탕으로 정답 선택지를"
              " 도출했습니다."
          ),
      })
      norm["questions"] = [first_q]
    norm["is_essay"] = False
    return norm

  if callable(spec.get("normalize_record")):
    norm = spec["normalize_record"](
        parsed, num_choices=0, num_correct=0, topic=topic
    )
    norm["is_essay"] = True
    return norm

  questions = parsed.get("questions") or []
  first_q = (
      dict(questions[0])
      if questions and isinstance(questions[0], dict)
      else {
          "number": 1,
          "stem": "제시된 내용을 바탕으로 알맞은 것을 서술하세요.",
          "box_example": None,
          "choices": [],
          "answer_indices": [],
          "answer": "",
          "explanation": "",
      }
  )
  stem_raw = str(first_q.get("stem") or "").strip()
  if stem_raw.startswith("{"):
    first_q["stem"] = (
        f"{topic or '제시된 주제'}의 핵심 쟁점과 한계점을 논리적으로 서술하세요."
    )
  first_q["number"] = 1
  first_q["choices"] = []
  first_q["num_choices"] = 0
  first_q["num_correct"] = 0
  first_q["answer_indices"] = []
  first_q["answer_index"] = None

  parsed["questions"] = [first_q]
  parsed["is_essay"] = True
  return parsed


def _collect_existing_history(
    dbms: QuizProjectDBMS, keyword: Optional[str], limit: int = 20
) -> Tuple[str, List[str], str]:
  """기존 퀴즈 요약, 기존 문항(stem) 목록, 이미 쓰인 원본 발췌(source_excerpts) 요약을 반환합니다."""
  summaries = dbms.get_existing_summaries(limit=limit)
  existing_ctx = "\n".join(f"- {s}" for s in summaries)

  records = dbms.select_quizzes(topic=keyword) if keyword else dbms.select_quizzes()
  if not records:
    records = dbms.select_quizzes()

  existing_stems: List[str] = []
  used_excerpts: List[str] = []
  for rec in records[-limit:]:
    for q in rec.get("questions") or []:
      if isinstance(q, dict):
        st = str(q.get("stem") or "").strip()
        if st and st not in existing_stems:
          existing_stems.append(st)
    for ex in rec.get("source_excerpts") or []:
      if isinstance(ex, dict):
        sm = str(ex.get("summary") or "").strip()
        src = str(ex.get("source") or "").strip()
        line = f"[{src}] {sm}" if src else sm
        if line and line not in used_excerpts:
          used_excerpts.append(line)

  used_excerpts_ctx = "\n".join(f"- {x[:200]}" for x in used_excerpts[-12:])
  return existing_ctx, existing_stems, used_excerpts_ctx


def is_too_similar_to_existing(
    candidate_text: str,
    existing_texts: List[str],
    threshold: float = 0.80,
) -> Tuple[bool, float]:
  """후보 문항이 기존 문항들과 의미적으로 지나치게 중복되는지 코사인 유사도로 검사합니다."""
  cand = (candidate_text or "").strip()
  pool = [
      str(t or "").strip()
      for t in (existing_texts or [])
      if str(t or "").strip()
  ]
  if not cand or not pool:
    return False, 0.0

  cand_vec = _embed_normalized([cand])[0]
  pool_vecs = _embed_normalized(pool)
  sims = pool_vecs @ cand_vec
  max_sim = float(np.max(sims)) if sims.size else 0.0
  return max_sim >= threshold, max_sim


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
  existing_stems: List[str]
  used_excerpts_ctx: str
  articles: List[Dict[str, Any]]
  relevance_ok: bool
  rag_feedback: str
  rag_retry_count: int
  aborted_no_articles: bool
  design_plan: Dict[str, Any]
  source_display_limit: int
  draft_quiz: Dict[str, Any]
  quality_report: Dict[str, Any]
  retry_count: int
  ls_project: str


def _articles_to_context(
    articles: List[Dict[str, Any]], max_items: Optional[int] = None
) -> str:
  if not articles:
    return "(저장된 기사 데이터 중 요청 주제와 직접 일치하는 기사가 부족합니다.)"
  sliced = articles[:max_items] if max_items and max_items > 0 else articles
  lines = []
  for idx, article in enumerate(sliced, 1):
    source = article.get("source") or article.get("press") or "언론"
    title = article.get("title") or ""
    facts = (
        article.get("facts")
        or article.get("summary")
        or article.get("content")
        or ""
    )
    stance = article.get("stance_type") or "중립"
    opinion = article.get("opinion") or ""
    lines.append(
        f"[{idx}] ({source}) {title}\n"
        f"    사실: {facts}\n"
        f"    관점({stance}): {opinion}"
    )
  return "\n".join(lines)


def _infer_type_with_llm(
    user_text: str,
    requested_type: Optional[str],
    llm_call_fn: Callable[[str, int], Optional[str]],
) -> str:
  available = load_all_quiz_types()
  if not available:
    return requested_type or "csat_passage_set"

  if requested_type and requested_type in available:
    return requested_type

  if _wants_multiple_choice(user_text):
    return _pick_mc_type_name(available, requested_type)

  if _wants_essay(user_text):
    return _pick_essay_type_name(available, requested_type)

  catalog = [
      {
          "name": k,
          "label": v.get("label", k),
          "allow_choices": bool(v.get("allow_choices")),
          "description": v.get("description", ""),
      }
      for k, v in available.items()
  ]
  prompt = (
      "사용자 요청에 가장 알맞은 문제 유형(name) 1개를 고르세요. JSON만 출력하세요.\n"
      '출력 형식: {"question_type": "유형name"}\n\n'
      f"사용자 요청: {user_text}\n"
      f"사용 가능한 유형 목록: {json.dumps(catalog, ensure_ascii=False)}"
  )
  parsed = _extract_json_obj(llm_call_fn(prompt, 300))
  picked = (parsed or {}).get("question_type")
  if picked in available:
    return picked
  return _pick_mc_type_name(available)


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
  type_spec = get_quiz_type_spec(resolved_type)
  max_n = int(
      type_spec.get("max_choices")
      or get_quiz_type_spec("csat_passage_set").get("max_choices")
      or 10
  )
  parsed_n, parsed_m = _parse_n_m_via_plugin(
      user_text, spec=type_spec, default_n=5, default_m=1
  )
  n_cnt, m_cnt = _clamp_choice_counts(
      num_choices, num_correct, parsed_n, parsed_m, max_n
  )

  topic_label = sanitize_topic_filename(keyword or f"random_{resolved_type}")
  ls_project = f"quiz-{topic_label}-{uuid.uuid4().hex[:4]}"

  os.environ["LANGCHAIN_TRACING_V2"] = "true"
  os.environ["LANGCHAIN_PROJECT"] = ls_project

  existing_ctx, existing_stems, used_excerpts_ctx = _collect_existing_history(
      dbms=dbms, keyword=keyword, limit=20
  )

  def _resolve_essay_and_choices_flag(
      state: QuizGenState, spec: Dict[str, Any]
  ) -> Tuple[bool, bool]:
    plan = state.get("design_plan") or {}
    if _wants_multiple_choice(state["user_text"]):
      return False, True
    if _wants_essay(state["user_text"]):
      return True, False
    is_essay = bool(plan.get("is_essay"))
    allow_choices = bool(spec.get("allow_choices")) and not is_essay
    return is_essay, allow_choices

  def node_fetch_rag(state: QuizGenState) -> Dict[str, Any]:
    query = state.get("rag_query") or state.get("keyword")
    articles = fetch_articles_fn(query, max(limit * 3, 12))
    return {"articles": articles or []}

  def node_judge_relevance(state: QuizGenState) -> Dict[str, Any]:
    target_topic = (state.get("keyword") or "").strip()
    articles = state.get("articles") or []
    retry_cnt = int(state.get("rag_retry_count") or 0)

    if not articles:
      return {
          "relevance_ok": False,
          "rag_retry_count": retry_cnt + 1,
          "rag_feedback": (
              f"수집한 RSS는 '{target_topic or '요청 주제'}'에 대한 문제를 만들기"
              " 부족합니다. 다른 카테고리 혹은 뉴스 데이터를 더 가져오시오."
          ),
      }

    if not target_topic:
      valid_articles = [
          a
          for a in articles
          if (a.get("title") or "").strip()
          and (
              a.get("facts")
              or a.get("summary")
              or a.get("content")
              or a.get("opinion")
              or ""
          ).strip()
      ]
      pool = valid_articles or articles
      random.shuffle(pool)
      picked = pool[: max(limit, 4)]
      titles = [str(a.get("title") or "") for a in picked if a.get("title")]
      stop_words = {"뉴스", "기사", "논란", "발표", "전망", "시장", "종합"}
      words = [
          w
          for t in titles
          for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", t)
          if w not in stop_words
      ]
      auto_kw = random.choice(words) if words else "시사_이슈"
      return {
          "keyword": auto_kw,
          "articles": picked,
          "relevance_ok": True,
          "rag_feedback": "",
      }

    kw_low = target_topic.lower()
    direct_matched = [
        a
        for a in articles
        if kw_low
        in (
            f"{a.get('title') or ''} {a.get('category') or ''}"
            f" {a.get('facts') or ''} {a.get('summary') or ''}"
            f" {a.get('opinion') or ''}"
        ).lower()
    ]

    compact = [
        {
            "idx": i,
            "title": str(a.get("title") or "")[:80],
            "category": str(a.get("category") or "")[:20],
            "summary": str(
                a.get("facts") or a.get("summary") or a.get("opinion") or ""
            )[:140],
        }
        for i, a in enumerate(articles[:10])
    ]

    prompt = (
        "하이브리드 RAG Top-K 검색 결과 중 요청 주제와 관련이 있거나 출제 소재로"
        " 활용할 수 있는 기사 idx를 고르세요.\n"
        f"요청 주제: {target_topic}\n"
        f"사용자 요청: {state['user_text'][:120]}\n"
        "명백한 무관 기사만 제외하고 JSON만 출력하세요: "
        '{"indices": [0, 1, 2], "reason": "짧은 사유"}\n'
        f"{json.dumps(compact, ensure_ascii=False)}"
    )

    parsed = _extract_json_obj(llm_call_fn(prompt, 400)) or {}
    raw_indices = (
        parsed.get("indices") if isinstance(parsed.get("indices"), list) else []
    )
    valid_indices: List[int] = []
    for idx in raw_indices:
      try:
        i_val = int(idx)
      except (TypeError, ValueError):
        continue
      if 0 <= i_val < len(articles[:10]) and i_val not in valid_indices:
        valid_indices.append(i_val)

    filtered = [articles[:10][i] for i in valid_indices]
    if not filtered and direct_matched:
      filtered = direct_matched[: max(limit * 2, 6)]

    if filtered:
      used_low = (state.get("used_excerpts_ctx") or "").lower()
      fresh_first = sorted(
          filtered,
          key=lambda a: (
              1
              if (
                  str(a.get("title") or "").strip().lower()[:30]
                  and str(a.get("title") or "").strip().lower()[:30]
                  in used_low
              )
              else 0
          ),
      )
      return {
          "articles": fresh_first,
          "relevance_ok": True,
          "rag_feedback": "",
      }

    return {
        "articles": [],
        "relevance_ok": False,
        "rag_retry_count": retry_cnt + 1,
        "rag_feedback": (
            f"수집한 RSS는 '{target_topic}'에 대한 문제를 만들기 부족합니다. "
            "다른 카테고리 혹은 뉴스 데이터를 더 가져오시오."
        ),
    }

  def node_rewrite_rag_query(state: QuizGenState) -> Dict[str, Any]:
    prev_q = state.get("rag_query") or state.get("keyword") or ""
    prompt = (
        "이전 하이브리드 RAG 검색에서 주제와 맞는 기사를 찾지 못했습니다.\n"
        "의미·키워드 검색에 쓸 핵심 대체 검색어 1개만 JSON으로 제안하세요.\n"
        '{"next_query": "대체검색어"}\n'
        f"원래 주제: {state.get('keyword')} | 이전 검색어: {prev_q}"
    )
    parsed = _extract_json_obj(llm_call_fn(prompt, 250)) or {}
    next_q = str(parsed.get("next_query") or "").strip() or None
    return {"rag_query": next_q}

  def route_after_relevance(state: QuizGenState) -> str:
    if state.get("relevance_ok") and (state.get("articles") or []):
      return "plan_design"
    if int(state.get("rag_retry_count") or 0) >= 2:
      return "abort"
    return "rewrite_rag"

  def node_abort_insufficient_data(state: QuizGenState) -> Dict[str, Any]:
    topic = state.get("keyword") or "해당 주제"
    msg = state.get("rag_feedback") or (
        f"수집한 RSS는 '{topic}'에 대한 문제를 만들기 부족합니다. "
        "다른 카테고리 혹은 뉴스 데이터를 더 가져오시오."
    )
    return {
        "articles": [],
        "aborted_no_articles": True,
        "draft_quiz": {
            "quiz_id": f"quiz_{uuid.uuid4().hex[:10]}",
            "topic": topic,
            "question_type": state.get("question_type") or resolved_type,
            "is_essay": False,
            "title": f"[{topic}] 문제 생성 중단",
            "passage": "",
            "source_excerpts": [],
            "questions": [],
            "rubric": [],
            "rag_warning": msg,
            "aborted_no_articles": True,
        },
        "quality_report": {"passed": False, "score": 0.0, "feedback": msg},
    }

  def node_plan_design(state: QuizGenState) -> Dict[str, Any]:
    articles = state.get("articles") or []
    q_type = state.get("question_type") or resolved_type
    spec = get_quiz_type_spec(q_type)

    if _wants_multiple_choice(state["user_text"]):
      is_essay = False
    elif _wants_essay(state["user_text"]):
      is_essay = True
    else:
      is_essay = bool(spec.get("is_essay")) or (
          not bool(spec.get("allow_choices"))
      )

    avail_cnt = len(articles)
    titles_preview = [
        f"{i + 1}. [{a.get('source') or a.get('press') or '언론'}] {a.get('title') or ''}"
        for i, a in enumerate(articles[:8])
    ]

    picked_axis = random.choice(_QUIZ_AXES)
    existing_ctx_str = state.get("existing_ctx") or ""
    for ax in _QUIZ_AXES:
      if ax["name"] not in existing_ctx_str:
        picked_axis = ax
        break

    prompt = (
        "당신은 출제 전략을 설계하는 LangGraph 설계자입니다.\n"
        "확보된 기사 수, 문제 유형, 그리고 이전 문제들과 차별화되는 출제 축을 반영해"
        " 반영할 기사 수(display_article_count)와 출제 각도(angle)를 JSON으로"
        " 결정하세요.\n"
        f"1. display_article_count는 1 이상 {max(1, min(avail_cnt, 8))} 이하"
        " 정수\n"
        f"2. 이번 출제 축: [{picked_axis['name']}] ({picked_axis['guide']})\n"
        "3. 기존 출제 이력과 같은 질문·관점·사례 반복을 피하도록 설계\n"
        'JSON만 출력: {"display_article_count": 3, "angle": "출제 각도",'
        ' "axis": "출제 축", "reasoning": "선택 이유"}\n\n'
        f"주제: {state.get('keyword')} | 유형: {q_type}"
        f" (is_essay={is_essay})\n"
        f"사용자 요청: {state['user_text']}\n"
        + (
            f"기존 출제 이력:\n{existing_ctx_str}\n"
            if existing_ctx_str
            else ""
        )
        + "확보된 기사 목록:\n"
        + ("\n".join(titles_preview) if titles_preview else "(없음)")
    )
    parsed = _extract_json_obj(llm_call_fn(prompt, 500)) or {}
    max_allowed = max(1, min(avail_cnt if avail_cnt > 0 else 1, 8))
    default_cnt = min(max_allowed, 2 if is_essay else 3)
    try:
      chosen_cnt = int(parsed.get("display_article_count"))
    except (TypeError, ValueError):
      chosen_cnt = default_cnt
    chosen_cnt = max(1, min(chosen_cnt, max_allowed))

    plan = {
        "is_essay": is_essay,
        "display_article_count": chosen_cnt,
        "axis": str(parsed.get("axis") or picked_axis["name"]),
        "angle": str(
            parsed.get("angle")
            or f"[{picked_axis['name']}] {picked_axis['guide']}"
        ),
        "reasoning": str(
            parsed.get("reasoning")
            or f"{picked_axis['name']} 축을 활용해 기존 문항과 다른 논점을 설계"
        ),
    }
    return {
        "design_plan": plan,
        "source_display_limit": chosen_cnt,
        "articles": articles[:chosen_cnt] if articles else [],
    }

  def node_generate_passage(state: QuizGenState) -> Dict[str, Any]:
    articles = state.get("articles") or []
    disp_limit = int(state.get("source_display_limit") or len(articles) or 1)
    flow_info = classify_articles_flow(articles[:disp_limit])
    ordered_articles = flow_info["ordered_articles"]
    ctx = _articles_to_context(ordered_articles, max_items=disp_limit)
    flow_outline = flow_info["flow_outline"]
    plan = state.get("design_plan") or {}
    rag_warn = state.get("rag_feedback") or ""
    used_excerpts = state.get("used_excerpts_ctx") or ""

    prompt = (
        f"주제 키워드: {state.get('keyword')}\n"
        f"출제 축 및 각도: {plan.get('axis', '')} / {plan.get('angle', '')}\n"
        f"임베딩 기반 글 전개 흐름: {flow_outline}\n"
        + (
            "이미 퀴즈에 활용된 발췌 요약(가능한 한 겹치지 않는 팩트·세부"
            f" 내용을 우선 반영):\n{used_excerpts}\n"
            if used_excerpts
            else ""
        )
        + f"참고 기사 ({disp_limit}건):\n{ctx}\n\n"
        "참고 기사 사실에 근거해 문제 출제용 배경 지문(passage), 세트"
        " 제목(title), "
        "기사 요약 목록(source_excerpts)을 작성하세요.\n"
        "제시된 전개 흐름에 맞춰 도입 → 전개/연결 → 마무리 순으로 논리적"
        " 연결어를 사용해 "
        "하나의 완결된 비문학 지문으로 구성하고, 중간에 문장을 끊지 마세요.\n"
        + (f"참고 알림: {rag_warn}\n" if rag_warn else "")
        + "\n반드시 아래 JSON 형식으로만 출력하세요:\n"
        "{\n"
        '  "title": "문제 세트 제목",\n'
        '  "passage": "문제 풀이에 필요한 배경 지문 또는 상황 설명",\n'
        '  "source_excerpts": [\n'
        '    {"source": "언론사/제목", "summary": "핵심 사실·관점 요약"}\n'
        "  ]\n"
        "}"
    )

    raw = llm_call_fn(prompt, 1600) or ""
    parsed = _extract_json_obj(raw) or {}
    excerpts = parsed.get("source_excerpts")
    if isinstance(excerpts, list) and len(excerpts) > disp_limit:
      excerpts = excerpts[:disp_limit]

    draft = dict(state.get("draft_quiz") or {})
    draft.update({
        "title": str(
            parsed.get("title") or f"{state.get('keyword')} 분석 과제"
        ),
        "passage": str(parsed.get("passage") or ctx[:1500]),
        "source_excerpts": excerpts if isinstance(excerpts, list) else [],
        "flow_outline": flow_outline,
    })
    return {"draft_quiz": draft}

  def node_generate_question(state: QuizGenState) -> Dict[str, Any]:
    draft = dict(state.get("draft_quiz") or {})
    passage = draft.get("passage") or ""
    q_type = state.get("question_type") or resolved_type
    spec = get_quiz_type_spec(q_type)
    is_essay, allow_choices = _resolve_essay_and_choices_flag(state, spec)
    plan = state.get("design_plan") or {}

    st_n, st_m = _clamp_choice_counts(
        state.get("num_choices"),
        state.get("num_correct"),
        n_cnt,
        m_cnt,
        max_n,
    )
    prev_feedback = (state.get("quality_report") or {}).get("feedback") or ""
    extra_rule = state.get("llm_rule") or ""
    rubric_list = state.get("custom_rubric") or DEFAULT_RUBRIC
    type_guide = (
        spec["build_guide"](rubric_list)
        if callable(spec.get("build_guide"))
        else ""
    )

    if is_essay:
      q_rule = (
          "서술형 1문항을 출제하세요. 단순 요약이 아닌 관점 비교·한계"
          " 비판·논증형 문항(stem)과 "
          "모범 답안 요지(answer), 해설(explanation), 채점 기준(rubric)을"
          " 작성하세요."
      )
    else:
      q_rule = (
          f"객관식 1문항의 질문 문장(stem)만 작성하세요. "
          "stem에는 배경 지문 내용을 다시 쓰거나 '1. ... 2. ...' / '① ... ②"
          " ...' 형태의 선택지를 절대 포함하지 마세요. "
          f"정답 번호 배열(answer_indices, 1~{st_n} 중 {st_m}개), "
          "정답 요약(answer), 해설(explanation), 채점 기준(rubric)을"
          " 작성하세요."
      )

    existing_section = (
        "이미 출제된 문항 및 요약(동일 질문·논점·발췌 반복 금지):\n"
        f"{state['existing_ctx']}\n"
        if state.get("existing_ctx")
        else ""
    )
    excerpts_avoid_section = (
        "기존 문제에 이미 쓰인 기사 발췌 요약(가능한 한 다른 팩트·근거를"
        f" 활용):\n{state['used_excerpts_ctx']}\n"
        if state.get("used_excerpts_ctx")
        else ""
    )

    prompt = (
        f"주제 키워드: {state.get('keyword')}\n"
        f"문제 유형: {q_type} ({spec.get('label', q_type)})\n"
        f"사용자 요청: {state['user_text']}\n"
        f"출제 축/각도: {plan.get('axis', '')} - {plan.get('angle', '')}\n"
        + (f"추가 규칙: {extra_rule}\n" if extra_rule else "")
        + existing_section
        + excerpts_avoid_section
        + f"배경 지문:\n{passage}\n\n"
        + (f"출제 지침: {type_guide}\n" if type_guide else "")
        + f"문항 작성 규칙: {q_rule}\n"
        + (
            f"이전 검증 피드백(반드시 반영): {prev_feedback}\n"
            if prev_feedback
            else ""
        )
        + "\n반드시 아래 JSON 형식으로만 출력하세요:\n"
        "{\n"
        '  "stem": "문항",\n'
        '  "box_example": null,\n'
        f'  "answer_indices": {[i for i in range(1, st_m + 1)]},\n'
        '  "answer": "정답 또는 모범 핵심 요지",\n'
        '  "explanation": "해설",\n'
        '  "rubric": ["채점 기준 1", "채점 기준 2"]\n'
        "}"
    )

    raw = llm_call_fn(prompt, 1600) or ""
    parsed = _extract_json_obj(raw) or {}

    cleaned_stem = _clean_stem_via_plugin(
        spec=spec,
        allow_choices=allow_choices,
        raw_stem=str(parsed.get("stem") or ""),
        passage=passage,
        topic=str(state.get("keyword") or ""),
    )

    ans_indices = parsed.get("answer_indices")
    if not isinstance(ans_indices, list) or not ans_indices:
      ans_indices = list(range(1, st_m + 1))

    prev_questions = draft.get("questions") or []
    prev_choices = (
        prev_questions[0].get("choices")
        if prev_questions and isinstance(prev_questions[0], dict)
        else []
    )

    q_item = {
        "number": 1,
        "stem": cleaned_stem,
        "box_example": parsed.get("box_example"),
        "choices": [] if is_essay else (prev_choices or []),
        "answer_indices": [] if is_essay else ans_indices,
        "answer": str(parsed.get("answer") or ""),
        "explanation": str(parsed.get("explanation") or ""),
    }

    rubric_out = parsed.get("rubric")
    if not isinstance(rubric_out, list) or not rubric_out:
      rubric_out = rubric_list

    draft["questions"] = [q_item]
    draft["rubric"] = rubric_out
    draft["is_essay"] = is_essay
    return {"draft_quiz": draft}

  def node_generate_choices(state: QuizGenState) -> Dict[str, Any]:
    draft = dict(state.get("draft_quiz") or {})
    q_type = state.get("question_type") or resolved_type
    spec = get_quiz_type_spec(q_type)
    plan = state.get("design_plan") or {}
    _, allow_choices = _resolve_essay_and_choices_flag(state, spec)

    st_n, st_m = _clamp_choice_counts(
        state.get("num_choices"),
        state.get("num_correct"),
        n_cnt,
        m_cnt,
        max_n,
    )

    questions = list(draft.get("questions") or [])
    first_q = (
        dict(questions[0])
        if questions and isinstance(questions[0], dict)
        else {"number": 1}
    )

    if not allow_choices:
      normalized_essay = _normalize_single_question_via_plugin(
          {**draft, "questions": [first_q]},
          spec=spec,
          allow_choices=False,
          num_choices=0,
          num_correct=0,
          topic=str(state.get("keyword") or ""),
      )
      articles = state.get("articles") or []
      disp_limit = int(state.get("source_display_limit") or len(articles) or 1)
      normalized_essay.update({
          "quiz_id": draft.get("quiz_id") or f"quiz_{uuid.uuid4().hex[:10]}",
          "topic": state["keyword"],
          "question_type": q_type,
          "is_essay": True,
          "source_display_limit": disp_limit,
          "design_plan": plan,
          "llm_rule": state.get("llm_rule") or "",
          "source_count": len(articles),
          "rag_relevance_ok": bool(state.get("relevance_ok")),
      })
      if state.get("rag_feedback"):
        normalized_essay["rag_warning"] = state["rag_feedback"]
      return {"draft_quiz": normalized_essay}

    prev_feedback = (state.get("quality_report") or {}).get("feedback") or ""
    prompt = (
        f"주제: {state.get('keyword')}\n"
        f"출제 축: {plan.get('axis', '')}\n"
        f"배경 지문:\n{draft.get('passage') or ''}\n\n"
        f"문항: {first_q.get('stem') or ''}\n"
        "정답 번호(answer_indices):"
        f" {first_q.get('answer_indices') or list(range(1, st_m + 1))}\n"
        f"정답 요약: {first_q.get('answer') or ''}\n"
        f"해설: {first_q.get('explanation') or ''}\n\n"
        f"객관식 선택지 {st_n}개를 작성하세요.\n"
        f"- 총 {st_n}개 선택지 중 {st_m}개는 지문/해설에 부합하는 정답 선택지,"
        " "
        f"나머지 {st_n - st_m}개는 지문 내용이나 인과관계를 비튼 구체적 오답"
        " 선택지로 작성하세요.\n"
        "- 선택지끼리 표현이나 의미가 중복되지 않도록 각각 다른 논점·근거를"
        " 담으세요.\n"
        "- '오답 선택지 1', '선택지 2', '플레이스홀더' 같은 임시/예시 문구는"
        " 절대 금지합니다.\n"
        + (
            f"- 이전 검증 피드백(반드시 반영): {prev_feedback}\n"
            if prev_feedback
            else ""
        )
        + "\n반드시 아래 JSON 형식으로만 출력하세요:\n"
        "{\n"
        '  "choices": ["① 구체적 선택지 내용", "② 구체적 선택지 내용"],\n'
        f'  "answer_indices": {[i for i in range(1, st_m + 1)]},\n'
        '  "explanation": "선택지 정오 근거를 포함한 해설"\n'
        "}"
    )

    raw = llm_call_fn(prompt, 1800) or ""
    parsed = _extract_json_obj(raw) or {}

    if isinstance(parsed.get("choices"), list) and parsed["choices"]:
      first_q["choices"] = parsed["choices"]
    if isinstance(parsed.get("answer_indices"), list) and parsed[
        "answer_indices"
    ]:
      first_q["answer_indices"] = parsed["answer_indices"]
    if parsed.get("explanation"):
      first_q["explanation"] = str(parsed["explanation"])

    draft["questions"] = [first_q]
    draft["is_essay"] = False

    normalized = _normalize_single_question_via_plugin(
        draft,
        spec=spec,
        allow_choices=True,
        num_choices=st_n,
        num_correct=st_m,
        topic=str(state.get("keyword") or ""),
    )

    articles = state.get("articles") or []
    disp_limit = int(state.get("source_display_limit") or len(articles) or 1)
    normalized.update({
        "quiz_id": draft.get("quiz_id") or f"quiz_{uuid.uuid4().hex[:10]}",
        "topic": state["keyword"],
        "question_type": q_type,
        "is_essay": False,
        "source_display_limit": disp_limit,
        "design_plan": plan,
        "llm_rule": state.get("llm_rule") or "",
        "source_count": len(articles),
        "rag_relevance_ok": bool(state.get("relevance_ok")),
    })
    if state.get("rag_feedback"):
      normalized["rag_warning"] = state["rag_feedback"]
    return {"draft_quiz": normalized}

  def node_verify(state: QuizGenState) -> Dict[str, Any]:
    quiz = state["draft_quiz"]
    q_type = quiz.get("question_type") or resolved_type
    spec = get_quiz_type_spec(q_type)
    next_retry = int(state.get("retry_count") or 0) + 1

    q_list = quiz.get("questions") or []
    first_q = q_list[0] if q_list and isinstance(q_list[0], dict) else {}
    stem_text = str(first_q.get("stem") or "").strip()

    too_sim, max_sim = is_too_similar_to_existing(
        candidate_text=stem_text,
        existing_texts=state.get("existing_stems") or [],
        threshold=0.80,
    )
    if too_sim:
      return {
          "quality_report": {
              "passed": False,
              "score": 0.35,
              "feedback": (
                  f"기존 문항과 의미 유사도({max_sim:.2f})가 높습니다. 다른"
                  " 출제 축이나 다른 팩트·근거를 활용해 새로운 문항으로 다시"
                  " 출제하세요."
              ),
          },
          "retry_count": next_retry,
      }

    validator = spec.get("validate")
    if callable(validator):
      plugin_ok, plugin_msg = validator(quiz)
      if not plugin_ok:
        return {
            "quality_report": {
                "passed": False,
                "score": 0.2,
                "feedback": plugin_msg,
            },
            "retry_count": next_retry,
        }

    sanitized_for_judge = {
        k: v for k, v in quiz.items() if k not in ("source_excerpts", "rubric")
    }
    judge_prompt = (
        "당신은 문항 논리 모순과 품질을 검증하는 심사관입니다. 0.0~1.0 점수와"
        " 판정을 JSON으로만 답하세요.\n"
        f"요청 주제: {state['keyword']} | 유형: {q_type} |"
        f" is_essay: {quiz.get('is_essay')}\n"
        f"사용자 원문: {state['user_text']}\n"
        "검증 항목:\n"
        "1. 서술형(is_essay=true)인데 선지형/객관식 문항처럼 구성되었거나"
        " 지문과 모순되는 점이 없는가?\n"
        "2. 객관식(is_essay=false)인데 선택지(choices)가 누락되거나 정답"
        " 번호가 비어있지 않은가?\n"
        "3. 문제 지문과 문항 요구사항이 서로 논리적으로 맞물리는가?\n"
        "검증 대상 JSON:\n"
        f"{json.dumps(sanitized_for_judge, ensure_ascii=False)[:3000]}\n\n"
        '출력 형식: {"passed": true, "score": 0.9, "feedback": "개선점 또는'
        ' 양호"}'
    )
    raw = llm_call_fn(judge_prompt, 500)
    judged = _extract_json_obj(raw)
    if not judged:
      has_q = bool(quiz.get("questions"))
      judged = {
          "passed": has_q,
          "score": 0.8 if has_q else 0.3,
          "feedback": "양호" if has_q else "문항을 보완하세요.",
      }
    try:
      score_val = float(
          judged.get("score") if judged.get("score") is not None else 0.5
      )
    except (TypeError, ValueError):
      score_val = 0.5
    judged["score"] = score_val
    judged["passed"] = bool(judged.get("passed")) and score_val >= 0.65
    return {
        "quality_report": judged,
        "retry_count": next_retry,
    }

  def route_after_verify(state: QuizGenState) -> str:
    if state["quality_report"].get("passed"):
      return "done"
    if int(state.get("retry_count") or 0) >= 2:
      return "done"
    return "retry"

  graph = StateGraph(QuizGenState)
  graph.add_node("fetch_rag", node_fetch_rag)
  graph.add_node("judge_relevance", node_judge_relevance)
  graph.add_node("rewrite_rag_query", node_rewrite_rag_query)
  graph.add_node("abort_no_data", node_abort_insufficient_data)
  graph.add_node("plan_design", node_plan_design)
  graph.add_node("generate_passage", node_generate_passage)
  graph.add_node("generate_question", node_generate_question)
  graph.add_node("generate_choices", node_generate_choices)
  graph.add_node("verify", node_verify)

  graph.set_entry_point("fetch_rag")
  graph.add_edge("fetch_rag", "judge_relevance")
  graph.add_conditional_edges(
      "judge_relevance",
      route_after_relevance,
      {
          "rewrite_rag": "rewrite_rag_query",
          "plan_design": "plan_design",
          "abort": "abort_no_data",
      },
  )
  graph.add_edge("rewrite_rag_query", "fetch_rag")
  graph.add_edge("abort_no_data", END)
  graph.add_edge("plan_design", "generate_passage")
  graph.add_edge("generate_passage", "generate_question")
  graph.add_edge("generate_question", "generate_choices")
  graph.add_edge("generate_choices", "verify")
  graph.add_conditional_edges(
      "verify", route_after_verify, {"retry": "generate_question", "done": END}
  )
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
      "existing_stems": existing_stems,
      "used_excerpts_ctx": used_excerpts_ctx,
      "articles": [],
      "relevance_ok": False,
      "rag_feedback": "",
      "rag_retry_count": 0,
      "aborted_no_articles": False,
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

  if final_state.get("aborted_no_articles") or quiz_record.get(
      "aborted_no_articles"
  ):
    return {
        "quiz_record": quiz_record,
        "saved_path": "",
        "ls_project": ls_project,
        "articles": [],
        "aborted": True,
    }

  saved_path = dbms.save_quiz(topic=quiz_record["topic"], quiz_record=quiz_record)
  return {
      "quiz_record": quiz_record,
      "saved_path": saved_path,
      "ls_project": ls_project,
      "articles": final_state.get("articles") or [],
      "aborted": False,
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
  available = load_all_quiz_types()
  topic = source_quiz.get("topic") or "general"
  topic_label = sanitize_topic_filename(topic)
  ls_project = (
      (source_quiz.get("langsmith") or {}).get("project_name")
      or f"quiz-{topic_label}-{uuid.uuid4().hex[:4]}"
  )

  want_essay = target_format == "essay" or (
      target_format is None and _wants_essay(user_text)
  )
  want_mc = target_format == "multiple_choice" or (
      target_format is None and _wants_multiple_choice(user_text)
  )

  if want_essay:
    resolved_type = _pick_essay_type_name(available, question_type)
    allow_choices = False
  elif want_mc:
    resolved_type = _pick_mc_type_name(available, question_type)
    allow_choices = True
  else:
    resolved_type = (
        question_type or source_quiz.get("question_type") or "csat_passage_set"
    )
    allow_choices = bool(
        get_quiz_type_spec(resolved_type).get("allow_choices")
    )

  spec = get_quiz_type_spec(resolved_type)
  max_n = int(
      spec.get("max_choices")
      or get_quiz_type_spec("csat_passage_set").get("max_choices")
      or 10
  )
  parsed_n, parsed_m = _parse_n_m_via_plugin(
      user_text, spec=spec, default_n=5, default_m=1
  )
  n_cnt, m_cnt = _clamp_choice_counts(
      num_choices, num_correct, parsed_n, parsed_m, max_n
  )

  source_summary = {
      "title": source_quiz.get("title"),
      "passage": source_quiz.get("passage"),
      "questions": source_quiz.get("questions"),
      "source_excerpts": source_quiz.get("source_excerpts"),
      "rubric": source_quiz.get("rubric"),
  }

  format_instruction = (
      "반드시 서술형 1문항으로 재탄생시키세요. choices는 반드시 빈 배열([])로"
      " 두고 객관식 선지를 절대 만들지 마세요."
      if not allow_choices
      else (
          f"반드시 객관식 1문항으로 재탄생시키세요. choices에 정확히 {n_cnt}개"
          " 선택지를 넣고, "
          f"{m_cnt}개 정답(answer_indices)과 {n_cnt - m_cnt}개 오답 함정 및"
          " 해설(explanation)을 작성하세요."
      )
  )

  prompt = (
      "기존 문제를 바탕으로 새로운 관점/형식의 퀴즈 1문항으로"
      " 재탄생시키세요. JSON만 출력하세요.\n"
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
      '      "choices": ["① 선택지1", "② 선택지2"],\n'
      '      "answer_indices": [1],\n'
      '      "answer": "정답/핵심요지",\n'
      '      "explanation": "해설"\n'
      "    }\n"
      "  ],\n"
      '  "rubric": ["채점 기준 1", "채점 기준 2"]\n'
      "}"
  )

  raw = llm_call_fn(prompt, 1800) or ""
  parsed = _extract_json_obj(raw) or {
      "title": f"{topic} 재구성 문항",
      "passage": source_quiz.get("passage") or "",
      "source_excerpts": source_quiz.get("source_excerpts") or [],
      "questions": [{
          "number": 1,
          "stem": (
              f"{topic}의 핵심 쟁점과 근거를 논리적으로 서술하세요."
              if not allow_choices
              else f"{topic} 관련 제시된 내용으로 가장 적절한 것은?"
          ),
          "box_example": None,
          "choices": [],
          "answer_indices": [1],
          "answer": "핵심 쟁점과 타당한 근거 제시",
          "explanation": "기존 문제의 쟁점과 근거를 바탕으로 재구성",
      }],
      "rubric": source_quiz.get("rubric")
      or ["핵심 근거의 정확성", "논리적 일관성"],
  }

  parsed = _normalize_single_question_via_plugin(
      parsed,
      spec=spec,
      allow_choices=allow_choices,
      num_choices=n_cnt,
      num_correct=m_cnt,
      topic=topic,
  )

  parsed.update({
      "quiz_id": f"quiz_{uuid.uuid4().hex[:10]}",
      "topic": topic,
      "question_type": resolved_type,
      "is_essay": not allow_choices,
      "source_display_limit": int(
          source_quiz.get("source_display_limit")
          or len(parsed.get("source_excerpts") or [])
          or 1
      ),
      "derived_from": {
          "quiz_id": source_quiz.get("quiz_id"),
          "topic": topic,
          "title": source_quiz.get("title"),
      },
      "quality_report": {
          "passed": True,
          "score": 0.9,
          "feedback": "기존 문제 기반 재탄생",
      },
      "langsmith": {"project_name": ls_project, "attempts": 1},
  })

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
      rec.get("source_display_limit")
      or len(rec.get("source_excerpts") or [])
      or 1
  )
  allow_choices = bool(spec.get("allow_choices")) and not bool(
      rec.get("is_essay")
  )

  saved_str = (
      os.path.relpath(result["saved_path"])
      if result.get("saved_path")
      else "미저장"
  )
  lines = [
      (
          f"📝 *[퀴즈 생성 완료]* `{rec.get('topic')}` | 유형:"
          f" `{rec.get('question_type')}` ({spec.get('label', '')})"
      ),
      (
          f"• 저장 파일: `{saved_str}` | LangSmith: `{result['ls_project']}`"
          f" (품질 {q_report.get('score', 0):.2f})"
      ),
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
        lines.append(
            f"• *{ex.get('source') or '기사'}*: {ex.get('summary') or ''}"
        )
    elif articles:
      lines.append("\n*참고 기사 자료*")
      for art in articles[:disp_limit]:
        title = art.get("title") or "제목 없음"
        body = (
            art.get("facts")
            or art.get("summary")
            or art.get("content")
            or art.get("opinion")
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
  """객관식/서술형/커스텀 유형 플러그인의 UI 빌더를 호출해 Slack 블록을 반환합니다."""
  rec = result["quiz_record"]
  spec = get_quiz_type_spec(rec.get("question_type") or "")
  allow_choices = bool(spec.get("allow_choices")) and not bool(
      rec.get("is_essay")
  )

  header_text = format_quiz_for_slack(
      result,
      articles=result.get("articles"),
      show_sources=show_sources,
      show_rubric=show_rubric,
  )

  blocks: List[Dict[str, Any]] = [{
      "type": "section",
      "text": {"type": "mrkdwn", "text": header_text[:2900]},
  }]

  quiz_id = rec["quiz_id"]
  if callable(spec.get("build_choice_blocks")) and allow_choices:
    blocks.extend(spec["build_choice_blocks"](rec))
  elif callable(spec.get("build_slack_blocks")):
    blocks.extend(spec["build_slack_blocks"](rec))
  elif allow_choices:
    block_builder = get_quiz_type_spec("csat_passage_set").get(
        "build_choice_blocks"
    )
    if callable(block_builder):
      blocks.extend(block_builder(rec))
  else:
    blocks.extend([
        {"type": "divider"},
        {
            "type": "actions",
            "block_id": f"quiz_essay_submit_block_{quiz_id}",
            "elements": [{
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "✍️ 서술형 답안 제출",
                    "emoji": True,
                },
                "style": "primary",
                "action_id": "quiz_essay_modal_open",
                "value": quiz_id,
            }],
        },
    ])

  return blocks