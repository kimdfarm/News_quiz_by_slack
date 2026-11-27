from typing import Any, Dict

TYPE_SPEC: Dict[str, Any] = {
    "name": "custom_rubric_essay",
    "label": "커스텀 루브릭 기반 코칭 서술형",
    "is_essay": True,
    "allow_choices": False,
    "description": "지정된 평가 기준(루브릭)에 맞춰 작성하는 코칭 서술형 문항",
    "build_guide": lambda rubric: (
        f"커스텀 루브릭 코칭 서술형 1문항을 만드세요. 적용할 채점 기준: {rubric}. "
        "답변 시 챙겨야 할 체크리스트와 힌트 단계도 함께 작성하세요. "
        "서술형이므로 choices는 반드시 빈 배열([])로 두고 객관식 선지를 절대 만들지 마세요."
    ),
    "validate": lambda quiz: (
        (False, "서술형 문제에 객관식 선지(choices)가 포함되어 있습니다.")
        if any(q.get("choices") for q in (quiz.get("questions") or []))
        else (True, "")
    ),
}