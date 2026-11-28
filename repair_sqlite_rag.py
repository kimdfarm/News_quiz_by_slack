import glob
import importlib.util
import os
import sqlite3
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def find_sqlite_paths(root_dir: str) -> list[str]:
    found: list[str] = []
    for current_root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in {"venv", ".venv", "__pycache__", ".git", "node_modules"}]
        if "storage_backend.py" in files:
            sb_path = os.path.join(current_root, "storage_backend.py")
            try:
                spec = importlib.util.spec_from_file_location("found_storage_backend", sb_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    for attr in ("DEFAULT_SQLITE_PATH", "SQLITE_DB_PATH", "DB_PATH"):
                        val = getattr(mod, attr, None)
                        if isinstance(val, str) and val.strip():
                            found.append(os.path.abspath(os.path.join(root_dir, val)))
            except Exception as e:
                print(f"[참고] {sb_path} 로드 중 일부 스킵: {e}")
    for ext in ("*.db", "*.sqlite", "*.sqlite3"):
        for path in glob.glob(os.path.join(root_dir, "**", ext), recursive=True):
            norm = os.path.normpath(path).lower()
            if any(skip in norm.split(os.sep) for skip in ("venv", ".venv", "__pycache__", ".git", "node_modules")):
                continue
            found.append(os.path.abspath(path))
    unique_paths: list[str] = []
    seen = set()
    for p in found:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)
    return unique_paths


def inspect_and_repair(db_path: str, auto_reset_rag: bool = True) -> None:
    print(f"\n=== 대상 DB: {db_path} ===")
    if not os.path.exists(db_path):
        print(" -> 파일이 존재하지 않습니다.")
        return
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("PRAGMA integrity_check;").fetchall()
        print(f" -> 무결성 검사: {rows}")
        if len(rows) == 1 and rows[0][0] == "ok":
            print(" -> 정상(ok)입니다.")
            return
        print(" -> 경고: SQLite 손상이 감지되었습니다.")
        if auto_reset_rag:
            print(" -> RAG 테이블(quiz_rag_vec, quiz_rag_fts, quiz_rag_docs) 초기화 시도...")
            for table in ("quiz_rag_vec", "quiz_rag_fts", "quiz_rag_docs"):
                try:
                    conn.execute(f"DROP TABLE IF EXISTS {table};")
                except sqlite3.DatabaseError as e:
                    print(f"    - {table} 삭제 오류: {e}")
            conn.commit()
            recheck = conn.execute("PRAGMA integrity_check;").fetchall()
            print(f" -> 초기화 후 재검사: {recheck}")
            if len(recheck) == 1 and recheck[0][0] == "ok":
                print(" -> RAG 테이블 초기화로 복구되었습니다.")
            else:
                print(" -> 기본 DB 테이블까지 손상되었습니다. 해당 .db(-wal/-shm 포함) 파일을 백업 후 재생성하세요.")
    except sqlite3.DatabaseError as e:
        print(f" -> SQLite 오류 발생: {e}")
        print(" -> DB 파일 자체 손상 가능성이 높습니다. 기존 파일을 백업 후 재생성하세요.")
    finally:
        conn.close()


if __name__ == "__main__":
    paths = find_sqlite_paths(ROOT_DIR)
    if not paths:
        print("대상 SQLite DB 파일을 찾지 못했습니다. (Neon/PostgreSQL 모드이거나 DB 생성 전일 수 있습니다.)")
    else:
        for p in paths:
            inspect_and_repair(p, auto_reset_rag=True)