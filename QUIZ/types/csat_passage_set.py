from typing import Any, Dict

TYPE_SPEC: Dict[str, Any] = {
    "name": "csat_passage_set",
    "label": "수능/비문학형 지문 + 2~3문항 세트",
    "is_essay": False,
    "allow_choices": True,
    "description": "비문학 지문과 <보기> 적용 문항이 포함된 객관식 세트",
    "build_guide": lambda rubric: (
        "수능/비문학형 지문 1개와 2~3개 객관식 문항 세트를 만드세요. "
        "<보기> 적용 문항을 최소 1개 포함하고, 각 문항마다 4~5개 선택지(choices), 정답, 해설을 작성하세요."
    ),
    "validate": lambda quiz: (
        (False, "객관식 세트 문항에 선택지(choices)가 비어 있습니다.")
        if not all(len(q.get("choices") or []) >= 2 for q in (quiz.get("questions") or []))
        else (True, "")
    ),
}