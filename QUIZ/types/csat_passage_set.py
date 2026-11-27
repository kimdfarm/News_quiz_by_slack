from typing import Any, Dict

TYPE_SPEC: Dict[str, Any] = {
    "name": "csat_passage_set",
    "label": "수능/비문학형 지문 + 1문항 객관식(N지선다·M정답)",
    "is_essay": False,
    "allow_choices": True,
    "description": "비문학 지문과 핵심 추론 객관식 1문항(기본 5지선다·1정답, N지선다·M정답 확장 지원)",
    "build_guide": lambda rubric: (
        "비문학 지문 1개와 객관식 1문항만 만드세요. "
        "요청된 N개 선택지(기본 5개)를 구성하되, M개(기본 1개)는 근거에 부합하는 정답으로 두고 "
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