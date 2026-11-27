from typing import Any, Dict

TYPE_SPEC: Dict[str, Any] = {
    "name": "fact_check_trap",
    "label": "팩트체크 & 함정 선지 판별형",
    "is_essay": True,
    "allow_choices": False,
    "description": "원문의 수치·주체·인과관계를 비튼 진술을 찾아 근거와 함께 교정하는 문항",
    "build_guide": lambda rubric: (
        "기사 원문의 수치/주체/인과관계를 교묘하게 비튼 진술(함정)을 포함한 팩트체크 판별 문항을 만드세요. "
        "틀린 부분, 실제 사실 근거, 수정 예시를 함께 작성하세요. choices는 빈 배열([])로 두세요."
    ),
    "validate": lambda quiz: (True, ""),
}