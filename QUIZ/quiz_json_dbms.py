import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


def sanitize_topic_filename(topic: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]+', "", topic or "").strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:60] or "general_quiz"


class QuizProjectDBMS:
    """PROJECT/<문제주제>.json 파일을 DBMS처럼 다루는 인코더/디코더 저장소"""

    def __init__(self, base_dir: str = "PROJECT"):
        self.base_dir = os.path.abspath(base_dir)
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

    def topic_file_path(self, topic: str) -> str:
        return os.path.join(self.base_dir, f"{sanitize_topic_filename(topic)}.json")

    def _topic_path(self, topic: str) -> str:
        return self.topic_file_path(topic)

    def list_topics(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for fn in sorted(os.listdir(self.base_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            fp = os.path.join(self.base_dir, fn)
            doc = self._read_json(fp)
            out.append({
                "file": fn,
                "topic": doc.get("topic"),
                "count": len(doc.get("records") or []),
                "updated_at": doc.get("updated_at"),
            })
        return out

    # --- 1. INSERT / UPSERT ---
    def save_quiz(
        self,
        topic: str,
        quiz_record: Dict[str, Any],
        thread_ts: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> str:
        file_path = self.topic_file_path(topic)
        now = datetime.now().isoformat()

        if os.path.exists(file_path):
            doc = self._read_json(file_path)
        else:
            doc = {
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

        existing_idx = next(
            (i for i, r in enumerate(doc.get("records") or []) if r.get("quiz_id") == quiz_id),
            None,
        )
        if existing_idx is not None:
            doc["records"][existing_idx] = quiz_record
        else:
            doc.setdefault("records", []).append(quiz_record)

        doc["updated_at"] = now
        self._atomic_write(file_path, doc)

        index_data = self._read_json(self.index_path)
        index_data.setdefault("quizzes", {})[quiz_id] = {
            "topic": topic,
            "file": os.path.basename(file_path),
            "question_type": quiz_record.get("question_type"),
            "langsmith_project": (quiz_record.get("langsmith") or {}).get("project_name"),
        }
        if thread_ts:
            index_data.setdefault("threads", {})[str(thread_ts)] = {
                "quiz_id": quiz_id,
                "topic": topic,
                "file": os.path.basename(file_path),
                "channel_id": channel_id,
            }
        self._atomic_write(self.index_path, index_data)
        return file_path

    def insert_quiz(
        self,
        topic: str,
        quiz_record: Dict[str, Any],
        thread_ts: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> str:
        return self.save_quiz(
            topic=topic,
            quiz_record=quiz_record,
            thread_ts=thread_ts,
            channel_id=channel_id,
        )

    # --- 2. SELECT ---
    def get_by_thread(self, thread_ts: str) -> Optional[Dict[str, Any]]:
        index_data = self._read_json(self.index_path)
        meta = (index_data.get("threads") or {}).get(str(thread_ts))
        if not meta:
            return None
        items = self.select_quizzes(topic=meta.get("topic"), quiz_id=meta.get("quiz_id"))
        return items[0] if items else None

    def get_by_quiz_id(self, quiz_id: str) -> Optional[Dict[str, Any]]:
        items = self.select_quizzes(quiz_id=quiz_id)
        return items[0] if items else None

    def select_quizzes(
        self,
        topic: Optional[str] = None,
        quiz_id: Optional[str] = None,
        question_type: Optional[str] = None,
        min_quality_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if topic:
            paths = [self.topic_file_path(topic)]
        else:
            paths = [
                os.path.join(self.base_dir, fn)
                for fn in sorted(os.listdir(self.base_dir))
                if fn.endswith(".json") and not fn.startswith("_")
            ]

        matched: List[Dict[str, Any]] = []
        for fp in paths:
            if not os.path.exists(fp):
                continue
            doc = self._read_json(fp)
            for rec in doc.get("records") or []:
                if quiz_id and rec.get("quiz_id") != quiz_id:
                    continue
                if question_type and rec.get("question_type") != question_type:
                    continue
                q_score = float(((rec.get("quality_report") or {}).get("score")) if rec.get("quality_report") is not None else 1.0)
                if q_score < min_quality_score:
                    continue
                matched.append(rec)
        return matched

    # --- 3. DELETE ---
    def delete_quiz(self, quiz_id: str) -> bool:
        index_data = self._read_json(self.index_path)
        meta = (index_data.get("quizzes") or {}).get(quiz_id)
        if not meta:
            return False

        fp = os.path.join(self.base_dir, meta["file"])
        if os.path.exists(fp):
            doc = self._read_json(fp)
            doc["records"] = [
                r for r in (doc.get("records") or []) if r.get("quiz_id") != quiz_id
            ]
            doc["updated_at"] = datetime.now().isoformat()
            self._atomic_write(fp, doc)

        (index_data.get("quizzes") or {}).pop(quiz_id, None)
        index_data["threads"] = {
            ts: info
            for ts, info in (index_data.get("threads") or {}).items()
            if info.get("quiz_id") != quiz_id
        }
        self._atomic_write(self.index_path, index_data)
        return True

    # --- 4. EXPORT / IMPORT ---
    def export_bundle(
        self,
        export_path: str,
        topics: Optional[List[str]] = None,
        question_types: Optional[List[str]] = None,
        template_style: str = "csat_2col",
    ) -> str:
        items: List[Dict[str, Any]] = []
        target_topics = topics or [t["topic"] for t in self.list_topics() if t.get("topic")]
        for t in target_topics:
            for rec in self.select_quizzes(topic=t):
                if question_types and rec.get("question_type") not in question_types:
                    continue
                items.append(rec)

        bundle = {
            "workbook_meta": {
                "style": template_style,
                "exported_at": datetime.now().isoformat(),
                "total_items": len(items),
            },
            "items": items,
        }
        self._atomic_write(export_path, bundle)
        return export_path

    def export_workbook_bundle(
        self,
        export_path: str,
        topics: Optional[List[str]] = None,
        question_types: Optional[List[str]] = None,
        template_style: str = "csat_2col",
    ) -> str:
        return self.export_bundle(
            export_path=export_path,
            topics=topics,
            question_types=question_types,
            template_style=template_style,
        )

    def import_bundle(self, external_json_path: str) -> int:
        raw = self._read_json(external_json_path)
        records = raw.get("records") or raw.get("items") or []
        fallback_topic = raw.get("topic") or os.path.splitext(os.path.basename(external_json_path))[0]
        count = 0
        for rec in records:
            if not isinstance(rec, dict) or not rec.get("quiz_id"):
                continue
            topic = rec.get("topic") or fallback_topic
            slack_info = rec.get("slack") or {}
            self.save_quiz(
                topic=topic,
                quiz_record=rec,
                thread_ts=slack_info.get("thread_ts"),
                channel_id=slack_info.get("channel_id"),
            )
            count += 1
        return count

    def import_from_json(self, external_json_path: str) -> int:
        return self.import_bundle(external_json_path)

    def get_existing_summaries(self, limit: int = 20) -> List[str]:
        summaries: List[str] = []
        for rec in self.select_quizzes():
            q_type = rec.get("question_type", "")
            topic = rec.get("topic", "")
            title = rec.get("title", "")
            summaries.append(f"[{topic} | {q_type}] {title}")
            if len(summaries) >= limit:
                break
        return summaries


QuizJsonDBMS = QuizProjectDBMS