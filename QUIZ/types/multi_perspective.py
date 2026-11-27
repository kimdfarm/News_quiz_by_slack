from typing import Any, Dict, List

TYPE_SPEC: Dict[str, Any] = {
    "name": "multi_perspective",
    "label": "다각도 언론 비교·논증 서술형",
    "is_essay": True,
    "allow_choices": False,
    "description": "서로 다른 시각이나 언론 보도의 핵심 차이를 비교·논증하는 서술형 문항",
    "build_guide": lambda rubric: (
        "다각도 언론 비교·논증 서술형 1문항을 만드세요. "
        "서로 다른 시각/언론의 핵심 차이를 묻고, 모범답안과 채점 기준(rubric)을 포함하세요. "
        "서술형이므로 choices는 반드시 빈 배열([])로 두고 객관식 선지를 절대 만들지 마세요."
    ),
    "validate": lambda quiz: (
        (False, "서술형 문제에 객관식 선지(choices)가 포함되어 있습니다.")
        if any(q.get("choices") for q in (quiz.get("questions") or []))
        else (True, "")
    ),
}