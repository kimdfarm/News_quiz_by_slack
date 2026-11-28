import hashlib
import json
import os
import pickle
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_EMBED_DIM = int(os.getenv("QUIZ_RAG_EMBED_DIM") or 384)
_DEFAULT_MODEL_NAME = os.getenv("QUIZ_RAG_EMBED_MODEL") or "paraphrase-multilingual-MiniLM-L12-v2"
_ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_COOKIE_RAG_FAISS_DIR = os.getenv(
    "QUIZ_RAG_FAISS_DIR",
    str(_ROOT_DIR / "cookie" / "faiss_quiz_rag"),
)
# 기존 코드 호환용 별칭
DEFAULT_COOKIE_RAG_SQLITE_PATH = DEFAULT_COOKIE_RAG_FAISS_DIR

_cached_embedder: Optional[Callable[[List[str]], List[List[float]]]] = None


def _get_default_embedder() -> Callable[[List[str]], List[List[float]]]:
    """SentenceTransformer가 있으면 다국어 임베딩을 사용하고, 없으면 경량 해시 벡터로 폴백합니다."""
    global _cached_embedder
    if _cached_embedder is not None:
        return _cached_embedder

    try:
        from sentence_transformers import SentenceTransformer

        device = "cpu" if os.getenv("CUDA_VISIBLE_DEVICES", "") == "" else None
        model = SentenceTransformer(_DEFAULT_MODEL_NAME, device=device) if device else SentenceTransformer(_DEFAULT_MODEL_NAME)

        def _encode(texts: List[str]) -> List[List[float]]:
            vecs = model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vecs]

        _cached_embedder = _encode
        return _cached_embedder
    except Exception:
        def _hash_encode(texts: List[str]) -> List[List[float]]:
            out: List[List[float]] = []
            for t in texts:
                vec = [0.0] * _EMBED_DIM
                tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", (t or "").lower())
                for tok in tokens:
                    h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
                    idx = h % _EMBED_DIM
                    sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
                    vec[idx] += sign
                norm = sum(x * x for x in vec) ** 0.5 or 1.0
                out.append([x / norm for x in vec])
            return out

        _cached_embedder = _hash_encode
        return _cached_embedder


def _article_doc_key(article: Dict[str, Any]) -> str:
    raw = (
        article.get("doc_key")
        or article.get("url")
        or f"{article.get('title') or ''}|{article.get('pub_dt') or ''}"
    )
    return hashlib.sha256(str(raw).strip().encode("utf-8")).hexdigest()[:32]


def _article_to_embed_text(article: Dict[str, Any]) -> str:
    parts = [
        str(article.get("category") or ""),
        str(article.get("title") or ""),
        str(article.get("facts") or article.get("summary") or article.get("content") or ""),
        str(article.get("opinion") or ""),
    ]
    return "\n".join(p.strip() for p in parts if p and p.strip())[:2000]


def _vector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32)
    arr = mat.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


def decide_index_sync_timing_with_llm(
    keyword: Optional[str],
    user_text: str,
    storage_mode: str,
    indexed_count: int,
    pending_articles: List[Dict[str, Any]],
    llm_call_fn: Callable[[str, int], Optional[str]],
) -> Dict[str, Any]:
    """새 기사 증분 갱신을 지금 수행할지, 생략할지, 별도 동기화로 돌릴지 LLM이 판단합니다."""
    pending_cnt = len(pending_articles)
    if pending_cnt == 0:
        return {"action": "skip_up_to_date", "reason": "미색인 신규 기사가 없습니다."}

    sample_titles = [str(a.get("title") or "")[:60] for a in pending_articles[:5]]
    prompt = (
        "당신은 하이브리드 RAG(벡터+키워드) 인덱스 갱신 스케줄러입니다. JSON만 출력하세요.\n"
        '출력 형식: {"action": "incremental_now | skip_up_to_date | defer_sync", "reason": "짧은 이유"}\n'
        "규칙:\n"
        "- incremental_now: 미색인 신규 기사가 적정량(1~50건)이고 현재 요청 주제 반영에 유리할 때\n"
        "- defer_sync: 미색인 신규 기사가 대량(50건 초과)이라 실시간 퀴즈 응답을 지연시키지 않고 별도 동기화로 넘길 때\n"
        "- skip_up_to_date: 기존 색인 데이터만으로 충분하거나 지금 갱신이 불필요할 때\n\n"
        f"백엔드: {storage_mode} | 기존 색인 수: {indexed_count} | 미색인 신규 수: {pending_cnt}\n"
        f"요청 키워드: {keyword or '전체'} | 사용자 요청: {user_text[:100]}\n"
        f"신규 기사 샘플: {json.dumps(sample_titles, ensure_ascii=False)}"
    )
    raw = llm_call_fn(prompt, 120) or ""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
        action = str((parsed or {}).get("action") or "").strip()
        if action in ("incremental_now", "skip_up_to_date", "defer_sync"):
            return {"action": action, "reason": str((parsed or {}).get("reason") or "")}
    except Exception:
        pass

    if pending_cnt > 50:
        return {"action": "defer_sync", "reason": "미색인 기사가 많아 별도 동기화로 분리"}
    return {"action": "incremental_now", "reason": "소량 신규 기사를 즉시 증분 색인"}


class PostgresHybridRAGStore:
    """Neon(PostgreSQL)용 pgvector + pgvectorscale(DiskANN) + FTS(tsvector) 스토어."""

    def __init__(self, embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None):
        self.embed_fn = embed_fn or _get_default_embedder()

    def ensure_schema(self, cur: Any) -> bool:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
            except Exception:
                pass

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS quiz_rag_chunks (
                    doc_key TEXT PRIMARY KEY,
                    source TEXT,
                    category TEXT,
                    title TEXT,
                    facts TEXT,
                    opinion TEXT,
                    stance_type TEXT,
                    url TEXT,
                    pub_dt TIMESTAMPTZ,
                    embedding vector({_EMBED_DIM}),
                    tsv tsvector GENERATED ALWAYS AS (
                        to_tsvector('simple', coalesce(category, '') || ' ' || coalesce(title, '') || ' ' || coalesce(facts, '') || ' ' || coalesce(opinion, ''))
                    ) STORED,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_quiz_rag_chunks_tsv ON quiz_rag_chunks USING GIN (tsv);")
            try:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_quiz_rag_chunks_diskann "
                    "ON quiz_rag_chunks USING diskann (embedding vector_cosine_ops);"
                )
            except Exception:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_quiz_rag_chunks_hnsw "
                    "ON quiz_rag_chunks USING hnsw (embedding vector_cosine_ops);"
                )
            return True
        except Exception as e:
            print(f"[PostgresHybridRAGStore 스키마 초기화 실패]: {e}")
            return False

    def get_status_and_pending(self, cur: Any, scan_limit: int = 120) -> Tuple[int, List[Dict[str, Any]]]:
        cur.execute("SELECT COUNT(*) FROM quiz_rag_chunks;")
        indexed_count = int((cur.fetchone() or [0])[0])

        cur.execute(
            """
            SELECT
                COALESCE(s.name, '기타언론') AS source,
                COALESCE(c.name, '종합') AS category,
                p.title,
                COALESCE(NULLIF(p.summary_facts, ''), i.representative_summary, i.issue_title, '') AS facts,
                COALESCE(p.author_opinion, '') AS opinion,
                COALESCE(p.stance_type, '중립') AS stance_type,
                p.url,
                COALESCE(p.published_at, i.published_at, p.created_at) AS pub_dt
            FROM news_perspectives p
            LEFT JOIN news_issues i ON i.id = p.issue_id
            LEFT JOIN categories c ON c.id = i.category_id
            LEFT JOIN sources s ON s.id = p.source_id
            ORDER BY pub_dt DESC NULLS LAST, p.id DESC
            LIMIT %s;
            """,
            (scan_limit,),
        )
        cols = ["source", "category", "title", "facts", "opinion", "stance_type", "url", "pub_dt"]
        recent = [dict(zip(cols, r)) for r in cur.fetchall()]
        if not recent:
            return indexed_count, []

        keys = [_article_doc_key(a) for a in recent]
        cur.execute("SELECT doc_key FROM quiz_rag_chunks WHERE doc_key = ANY(%s);", (keys,))
        existing = {r[0] for r in cur.fetchall()}
        pending = [a for a, k in zip(recent, keys) if k not in existing]
        return indexed_count, pending

    def upsert_articles(self, cur: Any, articles: List[Dict[str, Any]]) -> int:
        if not articles or not self.embed_fn:
            return 0
        texts = [_article_to_embed_text(a) for a in articles]
        vectors = self.embed_fn(texts)
        inserted = 0
        for art, vec in zip(articles, vectors):
            dkey = _article_doc_key(art)
            cur.execute(
                """
                INSERT INTO quiz_rag_chunks (
                    doc_key, source, category, title, facts, opinion, stance_type, url, pub_dt, embedding, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, NOW())
                ON CONFLICT (doc_key) DO UPDATE SET
                    facts = EXCLUDED.facts,
                    opinion = EXCLUDED.opinion,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW();
                """,
                (
                    dkey,
                    art.get("source") or "기타언론",
                    art.get("category") or "종합",
                    art.get("title") or "",
                    art.get("facts") or art.get("summary") or "",
                    art.get("opinion") or "",
                    art.get("stance_type") or "중립",
                    art.get("url") or "",
                    art.get("pub_dt"),
                    _vector_literal(vec),
                ),
            )
            inserted += 1
        return inserted

    def hybrid_search(
        self,
        cur: Any,
        query_text: str,
        keyword: Optional[str],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        if not self.embed_fn:
            return []
        q_str = f"{keyword or ''} {query_text or ''}".strip() or "최신 주요 뉴스"
        q_vec = _vector_literal(self.embed_fn([q_str])[0])
        kw_clean = (keyword or "").strip()

        cur.execute(
            """
            WITH vec_hits AS (
                SELECT
                    doc_key, source, category, title, facts, opinion, stance_type, url, pub_dt,
                    (1.0 - (embedding <=> %s::vector)) AS vec_score
                FROM quiz_rag_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            ),
            fts_hits AS (
                SELECT
                    doc_key,
                    ts_rank_cd(tsv, plainto_tsquery('simple', %s)) AS fts_score
                FROM quiz_rag_chunks
                WHERE %s <> '' AND tsv @@ plainto_tsquery('simple', %s)
                LIMIT %s
            )
            SELECT
                v.source, v.category, v.title, v.facts, v.opinion, v.stance_type, v.url, v.pub_dt,
                (v.vec_score * 0.72 + COALESCE(f.fts_score, 0.0) * 0.28) AS hybrid_score
            FROM vec_hits v
            LEFT JOIN fts_hits f ON f.doc_key = v.doc_key
            ORDER BY hybrid_score DESC, v.pub_dt DESC NULLS LAST
            LIMIT %s;
            """,
            (q_vec, q_vec, max(top_k * 3, 20), kw_clean, kw_clean, kw_clean, max(top_k * 3, 20), top_k),
        )
        cols = ["source", "category", "title", "facts", "opinion", "stance_type", "url", "pub_dt", "rag_score"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


class FAISSHybridRAGStore:
    """로컬 파일 기반 FAISS 벡터 + 키워드 하이브리드 스토어 (SQLite 가상 테이블 손상 없음)."""

    _io_lock = threading.RLock()

    def __init__(
        self,
        index_dir: str = DEFAULT_COOKIE_RAG_FAISS_DIR,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self.index_dir = Path(index_dir or DEFAULT_COOKIE_RAG_FAISS_DIR)
        self.embed_fn = embed_fn or _get_default_embedder()
        self.index_path = self.index_dir / "quiz_rag.index"
        self.meta_path = self.index_dir / "quiz_rag_meta.pkl"
        self._faiss_available = self._check_faiss()

    @staticmethod
    def _check_faiss() -> bool:
        try:
            import faiss  # noqa: F401
            return True
        except Exception:
            return False

    def _new_empty_index(self, dim: int) -> Any:
        if not self._faiss_available:
            return None
        import faiss
        return faiss.IndexFlatIP(dim)

    def _load_state(self) -> Tuple[Any, List[Dict[str, Any]], np.ndarray]:
        """(faiss_index_or_None, docs_list, embeddings_matrix)를 안전하게 로드합니다."""
        docs: List[Dict[str, Any]] = []
        mat = np.zeros((0, _EMBED_DIM), dtype=np.float32)

        if self.meta_path.is_file():
            try:
                with open(self.meta_path, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict):
                    raw_docs = payload.get("docs") or []
                    raw_vecs = payload.get("vectors")
                    if isinstance(raw_docs, list):
                        docs = [d for d in raw_docs if isinstance(d, dict)]
                    if raw_vecs is not None:
                        arr = np.asarray(raw_vecs, dtype=np.float32)
                        if arr.ndim == 2 and arr.shape[1] > 0:
                            mat = _normalize_matrix(arr)
            except Exception as e:
                print(f"[FAISS 메타데이터 로드 실패, 빈 상태로 재시작]: {e}")
                docs = []
                mat = np.zeros((0, _EMBED_DIM), dtype=np.float32)

        # 문서 수와 벡터 행 수 정합성 맞추기
        if len(docs) != len(mat):
            n = min(len(docs), len(mat))
            docs = docs[:n]
            mat = mat[:n]

        dim = mat.shape[1] if mat.ndim == 2 and mat.shape[1] > 0 else _EMBED_DIM
        faiss_idx = self._new_empty_index(dim)

        if self._faiss_available and faiss_idx is not None:
            loaded_from_disk = False
            if self.index_path.is_file() and len(mat) > 0:
                try:
                    import faiss
                    idx = faiss.read_index(str(self.index_path))
                    if idx.ntotal == len(mat) and idx.d == dim:
                        faiss_idx = idx
                        loaded_from_disk = True
                except Exception:
                    loaded_from_disk = False

            if not loaded_from_disk and len(mat) > 0:
                faiss_idx.add(_normalize_matrix(mat))

        return faiss_idx, docs, mat

    def _save_state(self, faiss_idx: Any, docs: List[Dict[str, Any]], mat: np.ndarray) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        norm_mat = _normalize_matrix(mat) if len(mat) > 0 else np.zeros((0, _EMBED_DIM), dtype=np.float32)

        tmp_meta = self.meta_path.with_suffix(".pkl.tmp")
        with open(tmp_meta, "wb") as f:
            pickle.dump({"docs": docs, "vectors": norm_mat}, f)
        os.replace(str(tmp_meta), str(self.meta_path))

        if self._faiss_available and faiss_idx is not None:
            import faiss
            tmp_idx = self.index_dir / "quiz_rag.index.tmp"
            faiss.write_index(faiss_idx, str(tmp_idx))
            os.replace(str(tmp_idx), str(self.index_path))

    def ensure_schema(self) -> None:
        with self._io_lock:
            faiss_idx, docs, mat = self._load_state()
            self._save_state(faiss_idx, docs, mat)

    def get_indexed_keys(self) -> set[str]:
        with self._io_lock:
            _, docs, _ = self._load_state()
            return {
                str(d.get("doc_key") or _article_doc_key(d))
                for d in docs
                if (d.get("doc_key") or _article_doc_key(d))
            }

    def upsert_articles(self, articles: List[Dict[str, Any]]) -> int:
        if not articles or not self.embed_fn:
            return 0
        with self._io_lock:
            faiss_idx, docs, mat = self._load_state()
            key_to_pos: Dict[str, int] = {
                str(d.get("doc_key") or _article_doc_key(d)): i for i, d in enumerate(docs)
            }

            to_add_arts: List[Dict[str, Any]] = []
            to_add_keys: List[str] = []
            to_update_pairs: List[Tuple[int, Dict[str, Any]]] = []

            for art in articles:
                dkey = _article_doc_key(art)
                record = {
                    "doc_key": dkey,
                    "source": str(art.get("source") or art.get("press") or "언론"),
                    "category": str(art.get("category") or "종합"),
                    "title": str(art.get("title") or ""),
                    "facts": str(art.get("facts") or art.get("summary") or art.get("content") or ""),
                    "opinion": str(art.get("opinion") or ""),
                    "stance_type": str(art.get("stance_type") or "중립"),
                    "url": str(art.get("url") or ""),
                    "pub_dt": str(art.get("pub_dt") or ""),
                }
                if dkey in key_to_pos:
                    to_update_pairs.append((key_to_pos[dkey], record))
                else:
                    to_add_arts.append(record)
                    to_add_keys.append(dkey)

            updated_cnt = 0
            if to_update_pairs:
                upd_texts = [_article_to_embed_text(rec) for _, rec in to_update_pairs]
                upd_vecs = _normalize_matrix(np.asarray(self.embed_fn(upd_texts), dtype=np.float32))
                for (pos, rec), vec in zip(to_update_pairs, upd_vecs):
                    docs[pos] = rec
                    if len(mat) > pos and mat.shape[1] == vec.shape[0]:
                        mat[pos] = vec
                    updated_cnt += 1

            added_cnt = 0
            if to_add_arts:
                add_texts = [_article_to_embed_text(rec) for rec in to_add_arts]
                add_vecs = _normalize_matrix(np.asarray(self.embed_fn(add_texts), dtype=np.float32))
                if len(mat) == 0:
                    mat = add_vecs
                elif mat.shape[1] == add_vecs.shape[1]:
                    mat = np.vstack([mat, add_vecs])
                docs.extend(to_add_arts)
                added_cnt = len(to_add_arts)

            if updated_cnt > 0 or added_cnt > 0:
                dim = mat.shape[1] if len(mat) > 0 else _EMBED_DIM
                new_idx = self._new_empty_index(dim)
                if self._faiss_available and new_idx is not None and len(mat) > 0:
                    new_idx.add(_normalize_matrix(mat))
                self._save_state(new_idx, docs, mat)

            return added_cnt + updated_cnt

    def hybrid_search(
        self,
        query_text: str,
        keyword: Optional[str],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        if not self.embed_fn:
            return []
        with self._io_lock:
            faiss_idx, docs, mat = self._load_state()
            if not docs or len(mat) == 0:
                return []

            q_str = f"{keyword or ''} {query_text or ''}".strip() or "최신 주요 뉴스"
            q_vec = _normalize_matrix(np.asarray(self.embed_fn([q_str]), dtype=np.float32))
            if q_vec.shape[1] != mat.shape[1]:
                return []

            vec_scores: Dict[int, float] = {}
            k_search = min(len(docs), max(top_k * 3, 20))
            if self._faiss_available and faiss_idx is not None and faiss_idx.ntotal == len(docs):
                sims, idxs = faiss_idx.search(q_vec, k_search)
                for pos, sim in zip(idxs[0], sims[0]):
                    if int(pos) >= 0:
                        # 코사인 유사도(-1~1)를 0~1 범위로 정규화
                        vec_scores[int(pos)] = max(0.0, min(1.0, (float(sim) + 1.0) * 0.5))
            else:
                sims_arr = (mat @ q_vec[0].T).reshape(-1)
                top_pos = np.argsort(-sims_arr)[:k_search]
                for pos in top_pos:
                    vec_scores[int(pos)] = max(0.0, min(1.0, (float(sims_arr[int(pos)]) + 1.0) * 0.5))

            # 키워드 매칭 점수 계산
            kw_scores: Dict[int, float] = {}
            raw_terms = re.findall(r"[가-힣A-Za-z0-9]{2,}", f"{keyword or ''} {query_text or ''}".lower())
            terms = list(dict.fromkeys(raw_terms))[:8]
            if terms:
                for idx, doc in enumerate(docs):
                    haystack = " ".join(
                        [
                            str(doc.get("category") or ""),
                            str(doc.get("title") or ""),
                            str(doc.get("facts") or ""),
                            str(doc.get("opinion") or ""),
                        ]
                    ).lower()
                    matched = sum(1 for t in terms if t in haystack)
                    if matched > 0:
                        kw_scores[idx] = matched / len(terms)

            candidate_ids = set(vec_scores.keys()) | set(kw_scores.keys())
            ranked = sorted(
                candidate_ids,
                key=lambda i: vec_scores.get(i, 0.0) * 0.72 + kw_scores.get(i, 0.0) * 0.28,
                reverse=True,
            )[:top_k]

            results: List[Dict[str, Any]] = []
            for idx in ranked:
                doc = dict(docs[idx])
                doc["rag_score"] = round(vec_scores.get(idx, 0.0) * 0.72 + kw_scores.get(idx, 0.0) * 0.28, 4)
                results.append(doc)
            return results


# 기존 코드 호환용 별칭
SQLiteHybridRAGStore = FAISSHybridRAGStore


def sync_rag_index_standalone(
    storage_mode: str,
    index_dir: str = DEFAULT_COOKIE_RAG_FAISS_DIR,
    candidate_articles: Optional[List[Dict[str, Any]]] = None,
    sqlite_path: Optional[str] = None,
) -> Dict[str, Any]:
    """별도 동기화(배치/스케줄러/수동 명령) 호출 시 미색인 기사를 일괄 반영합니다."""
    mode = (storage_mode or "sqlite").strip().lower()
    if mode in ("postgres", "postgresql", "neon") and os.getenv("DATABASE_URL"):
        from database import get_db_connection

        store = PostgresHybridRAGStore()
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            store.ensure_schema(cur)
            _, pending = store.get_status_and_pending(cur, scan_limit=300)
            added = store.upsert_articles(cur, pending)
            conn.commit()
            return {"backend": "postgres(pgvector+pgvectorscale)", "synced_count": added}
        finally:
            cur.close()
            conn.close()

    target_dir = sqlite_path or index_dir or DEFAULT_COOKIE_RAG_FAISS_DIR
    store = FAISSHybridRAGStore(index_dir=target_dir)
    store.ensure_schema()
    existing = store.get_indexed_keys()
    pending = [a for a in (candidate_articles or []) if _article_doc_key(a) not in existing]
    added = store.upsert_articles(pending)
    backend_label = "faiss+keyword" if store._faiss_available else "numpy-cosine+keyword"
    return {
        "backend": f"local({Path(target_dir).name}:{backend_label})",
        "synced_count": added,
    }
