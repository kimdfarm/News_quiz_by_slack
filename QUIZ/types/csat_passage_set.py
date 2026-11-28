import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from QUIZ.quiz_semantic_classifier import _embed_normalized

CIRCLE_NUMS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]

_PLACEHOLDER_MARKERS = (
    "지문 내용과 다른 오답 선택지",
    "오답 선택지",
    "선택지 1",
    "선택지 2",
    "placeholder",
)


def parse_n_m_from_text(
    user_text: str,
    default_n: int = 5,
    default_m: int = 1,
) -> Tuple[int, int]:
  """자연어에서 N지선다(N, 최대 10)와 M개 정답(M)을 추출합니다. (기본 N=5, M=1)"""
  txt = user_text or ""
  n_val = default_n
  m_val = default_m

  m_n = re.search(r"(\d{1,2})\s*지\s*선\s*다", txt)
  if m_n:
    n_val = int(m_n.group(1))

  m_ans = re.search(
      r"(?:정답\s*(\d{1,2})\s*개|(\d{1,2})\s*개\s*정답|복수\s*정답\s*(\d{1,2})?)",
      txt,
  )
  if m_ans:
    raw_m = m_ans.group(1) or m_ans.group(2) or m_ans.group(3)
    if raw_m:
      m_val = int(raw_m)
    else:
      m_val = max(2, default_m)
  elif any(
      k in txt
      for k in (
          "복수정답",
          "복수 정답",
          "다중정답",
          "다중 정답",
          "여러 개 정답",
          "모두 고르",
      )
  ):
    m_val = max(2, default_m)

  n_val = max(2, min(n_val, len(CIRCLE_NUMS)))
  m_val = max(1, min(m_val, n_val - 1))
  return n_val, m_val


def clean_mc_stem(raw_stem: str, passage: str = "", topic: str = "") -> str:
  """객관식 문항 stem에서 지문 복붙이나 선지 나열 줄을 제거하고 정제합니다."""
  stem = str(raw_stem or "").strip()
  if not stem:
    return f"{topic or '제시된 주제'} 관련 설명으로 가장 적절한 것은?"

  stem = re.split(r"\n\s*(?:[①-⑩]|(?:[1-9]|10)\.\s)", stem)[0].strip()
  if len(stem) > 220 and passage and passage[:60] in stem:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[?？])\s*", stem)
        if s.strip().endswith(("?", "？"))
    ]
    if sentences:
      stem = sentences[-1]
  return stem


def has_placeholder_choice(choices: List[Any]) -> bool:
  """임시/플레이스홀더 선지가 섞여 있는지 확인합니다."""
  for ch in choices or []:
    s = str(ch or "").strip().lower()
    if not s or any(m.lower() in s for m in _PLACEHOLDER_MARKERS):
      return True
  return False


def has_duplicate_choices(
    choices: List[str],
    threshold: float = 0.85,
) -> bool:
  """객관식 선택지들 사이에 의미적으로 지나치게 중복되는 선지가 있는지 검사합니다."""
  cleaned = [
      str(c or "").strip() for c in (choices or []) if str(c or "").strip()
  ]
  if len(cleaned) < 2:
    return False

  vecs = _embed_normalized(cleaned)
  sim_matrix = vecs @ vecs.T
  for i in range(len(cleaned)):
    for j in range(i + 1, len(cleaned)):
      if float(sim_matrix[i, j]) >= threshold:
        return True
  return False


def normalize_choices_n(raw_choices: List[Any], n_choices: int) -> List[str]:
  """선지 개수를 N개(최대 10개)로 맞추고 ①~⑩ 번호를 통일합니다."""
  limit_n = max(2, min(int(n_choices or 5), len(CIRCLE_NUMS)))
  items = list(raw_choices or [])[:limit_n]
  while len(items) < limit_n:
    items.append(f"지문 내용과 다른 오답 선택지 {len(items) + 1}")

  normalized: List[str] = []
  for idx, ch in enumerate(items):
    text = str(ch or "").strip()
    text = re.sub(
        r"^(?:[①②③④⑤⑥⑦⑧⑨⑩]|\d+[\.\)\:]|\(\d+\))\s*",
        "",
        text,
    ).strip()
    normalized.append(f"{CIRCLE_NUMS[idx]} {text}")
  return normalized


def normalize_csat_question_record(
    parsed: Dict[str, Any],
    num_choices: int = 5,
    num_correct: int = 1,
) -> Dict[str, Any]:
  """1문항 고정 + N지선다 선지 정규화 + M개 정답 인덱스(answer_indices) 확정"""
  n_cnt = max(2, min(int(num_choices or 5), len(CIRCLE_NUMS)))
  m_cnt = max(1, min(int(num_correct or 1), n_cnt - 1))

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

  norm_choices = normalize_choices_n(
      first_q.get("choices") or [], n_choices=n_cnt
  )
  first_q["choices"] = norm_choices
  first_q["num_choices"] = n_cnt
  first_q["num_correct"] = m_cnt

  ans_indices: List[int] = []
  raw_indices = first_q.get("answer_indices")
  if isinstance(raw_indices, list):
    for x in raw_indices:
      try:
        xi = int(x)
        if 1 <= xi <= n_cnt and xi not in ans_indices:
          ans_indices.append(xi)
      except Exception:
        continue

  if not ans_indices:
    raw_ans = str(first_q.get("answer") or "")
    for i, sym in enumerate(CIRCLE_NUMS[:n_cnt], 1):
      if sym in raw_ans or re.search(rf"(?:^|[^\d]){i}(?:번|[^\d]|$)", raw_ans):
        if i not in ans_indices:
          ans_indices.append(i)

  for fallback_i in range(1, n_cnt + 1):
    if len(ans_indices) >= m_cnt:
      break
    if fallback_i not in ans_indices:
      ans_indices.append(fallback_i)

  ans_indices = sorted(ans_indices[:m_cnt])
  first_q["answer_indices"] = ans_indices
  first_q["answer_index"] = ans_indices[0] if ans_indices else 1
  first_q["answer"] = ", ".join(norm_choices[i - 1] for i in ans_indices)

  parsed["questions"] = [first_q]
  parsed["is_essay"] = False
  return parsed


def extract_choice_numbers(user_answer: str, max_choices: int) -> List[int]:
  """사용자 답안 문자열에서 ①~⑩ 또는 1~10 번호를 추출합니다."""
  limit_n = max(1, min(int(max_choices or len(CIRCLE_NUMS)), len(CIRCLE_NUMS)))
  s = (user_answer or "").strip()
  found: List[int] = []
  for idx, sym in enumerate(CIRCLE_NUMS[:limit_n], 1):
    if sym in s and idx not in found:
      found.append(idx)

  for m in re.finditer(r"(?:^|[^\d])([1-9]|10)(?:\s*번|[^\d]|$)", s):
    num = int(m.group(1))
    if 1 <= num <= limit_n and num not in found:
      found.append(num)

  return sorted(found)


def grade_mc_answers(
    quiz_record: Dict[str, Any],
    selected_indices: Optional[List[int]],
    raw_user_text: str,
    user_id: str,
) -> str:
  """객관식 문항 답안을 결정론적으로 채점해 Slack 메시지를 반환합니다."""
  q = (quiz_record.get("questions") or [{}])[0]
  choices = q.get("choices") or []
  n_cnt = len(choices)
  raw_ans_indices = q.get("answer_indices")

  if isinstance(raw_ans_indices, list) and raw_ans_indices:
    ans_indices = sorted(
        {int(x) for x in raw_ans_indices if 1 <= int(x) <= n_cnt}
    )
  elif q.get("answer_index"):
    ans_indices = [int(q["answer_index"])]
  else:
    ans_indices = [1] if n_cnt >= 1 else []

  expected_m = int(q.get("num_correct") or len(ans_indices) or 1)
  chosen_list = (
      sorted({int(x) for x in selected_indices if 1 <= int(x) <= n_cnt})
      if selected_indices is not None
      else extract_choice_numbers(raw_user_text, max_choices=n_cnt)
  )

  if not chosen_list:
    return (
        f"<@{user_id}> 선택하신 번호를 인식하지 못했어요. "
        f"`1번`~`{n_cnt}번`({expected_m}개)으로 입력하거나 위 UI에서 선택 후"
        " 제출해 주세요."
    )

  chosen_labels = ", ".join(choices[i - 1] for i in chosen_list)
  ans_labels = ", ".join(choices[i - 1] for i in ans_indices)
  is_correct = chosen_list == ans_indices

  if is_correct:
    verdict = "⭕ *정답입니다!*"
  elif len(chosen_list) != expected_m:
    verdict = (
        f"❌ *오답입니다.* (이 문항은 *{expected_m}개* 정답인데"
        f" *{len(chosen_list)}개*를 선택하셨습니다.)"
    )
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


def validate_csat_quiz(quiz: Dict[str, Any]) -> Tuple[bool, str]:
  """객관식 문항 수, 선택지 구성, 플레이스홀더 및 선지 간 의미 중복을 검증합니다."""
  questions = quiz.get("questions") or []
  if len(questions) != 1:
    return False, "객관식 문항은 정확히 1문항이어야 합니다."

  first_q = questions[0] if isinstance(questions[0], dict) else {}
  choices = first_q.get("choices") or []
  if len(choices) < 2:
    return False, "선택지(choices)가 최소 2개 이상 필요합니다."

  expected_n = int(first_q.get("num_choices") or 0)
  if expected_n >= 2 and len(choices) != expected_n:
    return (
        False,
        f"객관식 선택지 {expected_n}개를 모두 채워 다시 작성하세요.",
    )

  if has_placeholder_choice(choices):
    return (
        False,
        "임시/플레이스홀더 문구 없이 모든 선택지를 구체적인 내용으로 다시"
        " 작성하세요.",
    )

  if has_duplicate_choices(choices, threshold=0.85):
    return (
        False,
        "선택지 간 의미 중복이 높습니다. 각 선택지가 서로 다른 논점이나"
        " 오답 근거를 갖도록 다시 작성하세요.",
    )

  return True, ""


def build_csat_choice_blocks(quiz_record: Dict[str, Any]) -> List[Dict[str, Any]]:
  """csat_passage_set 전용 Slack 선택 UI(라디오/체크박스 + 제출 버튼) 블록을 생성합니다."""
  q = (quiz_record.get("questions") or [{}])[0]
  choices = (q.get("choices") or [])[: len(CIRCLE_NUMS)]
  if not choices:
    return []

  n_cnt = len(choices)
  m_cnt = int(q.get("num_correct") or len(q.get("answer_indices") or []) or 1)
  quiz_id = quiz_record["quiz_id"]

  options = [
      {
          "text": {"type": "plain_text", "text": ch[:75], "emoji": True},
          "value": f"{idx}|{ch[:60]}",
      }
      for idx, ch in enumerate(choices, 1)
  ]
  accessory_type = "checkboxes" if m_cnt > 1 else "radio_buttons"

  return [
      {"type": "divider"},
      {
          "type": "section",
          "block_id": f"quiz_select_block_{quiz_id}",
          "text": {
              "type": "mrkdwn",
              "text": f"*답안 선택 ({n_cnt}지선다 · *{m_cnt}개* 선택 후 제출)*",
          },
          "accessory": {
              "type": accessory_type,
              "action_id": "quiz_choice_select",
              "options": options,
          },
      },
      {
          "type": "actions",
          "block_id": f"quiz_submit_block_{quiz_id}",
          "elements": [{
              "type": "button",
              "text": {
                  "type": "plain_text",
                  "text": "✅ 답안 제출하기",
                  "emoji": True,
              },
              "style": "primary",
              "action_id": "quiz_choice_submit",
              "value": quiz_id,
          }],
      },
  ]


TYPE_SPEC: Dict[str, Any] = {
    "name": "csat_passage_set",
    "label": "수능/비문학형 지문 + 1문항 객관식(N지선다·M정답)",
    "is_essay": False,
    "allow_choices": True,
    "max_choices": len(CIRCLE_NUMS),
    "circle_nums": CIRCLE_NUMS,
    "parse_n_m": parse_n_m_from_text,
    "clean_stem": clean_mc_stem,
    "has_duplicate_choices": has_duplicate_choices,
    "normalize_record": normalize_csat_question_record,
    "build_choice_blocks": build_csat_choice_blocks,
    "extract_choice_numbers": extract_choice_numbers,
    "grade_mc": grade_mc_answers,
    "description": (
        "비문학 지문과 핵심 추론 객관식 1문항(기본 5지선다·1정답, 최대"
        " 10지선다·M정답 지원)"
    ),
    "build_guide": lambda rubric: (
        "비문학 지문 1개와 객관식 1문항만 만드세요. 요청된 N개 선택지(기본"
        " 5개, 최대 10개)를 구성하되, M개(기본 1개)는 근거에 부합하는"
        " 정답으로 두고 나머지 N-M개는 지문과 미묘하게 어긋나는 함정/오답"
        " 선지로 설계하세요. 선택지 간 의미 중복을 피하고, 정답 번호"
        " 배열(answer_indices, 1부터 시작)과 해설(explanation)을 명확히"
        " 작성하세요."
    ),
    "validate": validate_csat_quiz,
}