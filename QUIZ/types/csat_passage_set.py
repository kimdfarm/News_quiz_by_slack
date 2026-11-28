import re
from typing import Any, Dict, List, Optional, Tuple

CIRCLE_NUMS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]


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
        for k in ("복수정답", "복수 정답", "다중정답", "다중 정답", "여러 개 정답", "모두 고르")
    ):
        m_val = max(2, default_m)

    n_val = max(2, min(n_val, len(CIRCLE_NUMS)))
    m_val = max(1, min(m_val, n_val - 1))
    return n_val, m_val


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

    norm_choices = normalize_choices_n(first_q.get("choices") or [], n_choices=n_cnt)
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
            "text": {"type": "mrkdwn", "text": f"*답안 선택 ({n_cnt}지선다 · *{m_cnt}개* 선택 후 제출)*"},
            "accessory": {
                "type": accessory_type,
                "action_id": "quiz_choice_select",
                "options": options,
            },
        },
        {
            "type": "actions",
            "block_id": f"quiz_submit_block_{quiz_id}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ 답안 제출하기", "emoji": True},
                    "style": "primary",
                    "action_id": "quiz_choice_submit",
                    "value": quiz_id,
                }
            ],
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
    "normalize_record": normalize_csat_question_record,
    "build_choice_blocks": build_csat_choice_blocks,
    "description": "비문학 지문과 핵심 추론 객관식 1문항(기본 5지선다·1정답, 최대 10지선다·M정답 지원)",
    "build_guide": lambda rubric: (
        "비문학 지문 1개와 객관식 1문항만 만드세요. "
        "요청된 N개 선택지(기본 5개, 최대 10개)를 구성하되, M개(기본 1개)는 근거에 부합하는 정답으로 두고 "
        "나머지 N-M개는 지문과 미묘하게 어긋나는 함정/오답 선지로 설계하세요. "
        "정답 번호 배열(answer_indices, 1부터 시작)과 해설(explanation)을 명확히 작성하세요."
    ),
    "validate": lambda quiz: (
        (False, "객관식 문항은 정확히 1문항이어야 합니다.")
        if len(quiz.get("questions") or []) != 1
        else (
            (False, "선택지(choices)가 최소 2개 이상 필요합니다.")
            if len((quiz.get("questions") or [{}])[0].get("choices") or []) < 2
            else (True, "")
        )
    ),
}