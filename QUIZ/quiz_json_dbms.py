import os
import re
import json
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional


def sanitize_topic_filename(topic: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]+', "", topic).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:60] or "general_quiz"


class QuizJsonDBMS:
    """
    PROJECT/<문제주제>.json 파일을 SQLite DBMS처럼 관리하는 인코더/디코더 엔진
    """

    def __init__(self, base_dir: str = "PROJECT"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.index_path = os.path.join(self.base_dir, "_index.json")
        if not os.path.exists(self.index_path):
            self._atomic_write(self.index_path, {"threads": {}, "quizzes": {}})

    # --- 저수준 원자적 입출력 (Encoder / Decoder) ---
    def _atomic_write(self, path: str, payload: Dict[str, Any]) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _read_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _topic_path(self, topic: str) -> str:
        return os.path.join(self.base_dir, f"{sanitize_topic_filename(topic)}.json")

    # --- 1. INSERT / UPSERT (문제 저장 + LangSmith 메타데이터 기록) ---
    def insert_quiz(
        self,
        topic: str,
        quiz_record: Dict[str, Any],
        thread_ts: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> str:
        file_path = self._topic_path(topic)
        now = datetime.now().isoformat()

        if os.path.exists(file_path):
            db_doc = self._read_json(file_path)
        else:
            db_doc = {
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

        # 동일 quiz_id가 있으면 덮어쓰고(UPSERT), 없으면 추가
        existing_idx = next(
            (i for i, r in enumerate(db_doc["records"]) if r["quiz_id"] == quiz_id),
            None,
        )
        if existing_idx is not None:
            db_doc["records"][existing_idx] = quiz_record
        else:
            db_doc["records"].append(quiz_record)

        db_doc["updated_at"] = now
        self._atomic_write(file_path, db_doc)

        # 글로벌 인덱스 갱신 (스레드 답안 채점 시 O(1) 조회용)
        index_data = self._read_json(self.index_path)
        index_data["quizzes"][quiz_id] = {
            "topic": topic,
            "file": os.path.basename(file_path),
            "question_type": quiz_record.get("question_type"),
            "langsmith_project": quiz_record.get("langsmith", {}).get("project_name"),
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

    # --- 2. SELECT (주제별 / 유형별 / 스레드별 디코딩 조회) ---
    def get_by_thread(self, thread_ts: str) -> Optional[Dict[str, Any]]:
        index_data = self._read_json(self.index_path)
        meta = index_data["threads"].get(thread_ts)
        if not meta:
            return None
        return self.get_by_quiz_id(meta["quiz_id"])

    def get_by_quiz_id(self, quiz_id: str) -> Optional[Dict[str, Any]]:
        index_data = self._read_json(self.index_path)
        meta = index_data["quizzes"].get(quiz_id)
        if not meta:
            return None
        file_path = os.path.join(self.base_dir, meta["file"])
        if not os.path.exists(file_path):
            return None
        db_doc = self._read_json(file_path)
        return next((r for r in db_doc["records"] if r["quiz_id"] == quiz_id), None)

    def select_quizzes(
        self,
        topic: Optional[str] = None,
        question_type: Optional[str] = None,
        min_quality_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        files = (
            [self._topic_path(topic)]
            if topic
            else [
                os.path.join(self.base_dir, fn)
                for fn in os.listdir(self.base_dir)
                if fn.endswith(".json") and not fn.startswith("_")
            ]
        )
        for fp in files:
            if not os.path.exists(fp):
                continue
            doc = self._read_json(fp)
            for rec in doc.get("records", []):
                q_score = rec.get("quality_report", {}).get("score", 1.0)
                if question_type and rec.get("question_type") != question_type:
                    continue
                if q_score < min_quality_score:
                    continue
                results.append(rec)
        return results

    # --- 3. DELETE (특정 문항 삭제 및 인덱스 정리) ---
    def delete_quiz(self, quiz_id: str) -> bool:
        index_data = self._read_json(self.index_path)
        meta = index_data["quizzes"].get(quiz_id)
        if not meta:
            return False

        file_path = os.path.join(self.base_dir, meta["file"])
        if os.path.exists(file_path):
            doc = self._read_json(file_path)
            doc["records"] = [r for r in doc["records"] if r["quiz_id"] != quiz_id]
            doc["updated_at"] = datetime.now().isoformat()
            self._atomic_write(file_path, doc)

        index_data["quizzes"].pop(quiz_id, None)
        index_data["threads"] = {
            ts: info
            for ts, info in index_data["threads"].items()
            if info["quiz_id"] != quiz_id
        }
        self._atomic_write(self.index_path, index_data)
        return True

    # --- 4. EXPORT / DOWNLOAD (수능형·문제집 PDF 생성기용 번들 디코딩 추출) ---
    def export_workbook_bundle(
        self,
        export_path: str,
        topics: Optional[List[str]] = None,
        question_types: Optional[List[str]] = None,
        template_style: str = "csat_2col",
    ) -> str:
        selected: List[Dict[str, Any]] = []
        target_topics = topics or [
            fn[:-5]
            for fn in os.listdir(self.base_dir)
            if fn.endswith(".json") and not fn.startswith("_")
        ]
        for t in target_topics:
            for rec in self.select_quizzes(topic=t):
                if question_types and rec.get("question_type") not in question_types:
                    continue
                selected.append(rec)

        bundle = {
            "workbook_meta": {
                "style": template_style,  # csat_2col(수능형) | standard_workbook(일반문제집)
                "exported_at": datetime.now().isoformat(),
                "total_items": len(selected),
            },
            "items": selected,
        }
        self._atomic_write(export_path, bundle)
        return export_path

    # --- 5. IMPORT / LOAD (외부 JSON 문제 파일 검증 후 복원/병합) ---
    def import_from_json(self, external_json_path: str) -> int:
        raw = self._read_json(external_json_path)
        # 단일 주제 DB 파일 형태 또는 export_workbook_bundle 형태 모두 디코딩 지원
        items = raw.get("records") or raw.get("items") or []
        topic = raw.get("topic") or os.path.splitext(os.path.basename(external_json_path))[0]

        imported_count = 0
        for item in items:
            if "quiz_id" not in item or "question_type" not in item:
                continue
            item_topic = item.get("topic") or topic
            thread_ts = item.get("slack", {}).get("thread_ts")
            channel_id = item.get("slack", {}).get("channel_id")
            self.insert_quiz(
                topic=item_topic,
                quiz_record=item,
                thread_ts=thread_ts,
                channel_id=channel_id,
            )
            imported_count += 1
        return imported_count