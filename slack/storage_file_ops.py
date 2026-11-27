import json
import os
import tempfile
import threading
from typing import Any, Dict, List

import requests

from slack.llm_router import (
    clean_str_arg,
    extract_delete_args_with_llm,
    resolve_backend_value_with_llm,
    resolve_positive_int_with_llm,
)
from storage_backend import DEFAULT_SQLITE_PATH, import_sqlite_file_data

_IMPORT_LOCK = threading.Lock()


def download_slack_file(file_info: Dict[str, Any]) -> str:
    url = file_info.get("url_private_download") or file_info.get("url_private")
    if not url:
        raise ValueError("Slack 파일 다운로드 URL이 없어요.")

    bot_token = (os.environ.get("SLACK_BOT_TOKEN") or "").strip()
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {bot_token}"},
        timeout=30,
    )
    resp.raise_for_status()

    fd, tmp_path = tempfile.mkstemp(prefix="slack_import_", suffix=".sqlite")
    try:
        with os.fdopen(fd, "wb") as tmp_f:
            tmp_f.write(resp.content)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return tmp_path


def process_slack_files(attached_files: List[Dict[str, Any]], say: Any) -> None:
    if not attached_files:
        return

    with _IMPORT_LOCK:
        for file_info in attached_files:
            orig_name = file_info.get("name") or "uploaded_file"
            tmp_path = None
            try:
                tmp_path = download_slack_file(file_info)
                result = import_sqlite_file_data(
                    uploaded_sqlite_path=tmp_path,
                    target_sqlite_path=DEFAULT_SQLITE_PATH,
                )
            except Exception as e:
                say(text=f"❌ `{orig_name}` 파일을 가져오는 중 실패했어요: {e}")
                continue
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            status = result.get("status")
            if status == "ok":
                say(
                    text=(
                        f"📥 *SQLite 데이터 복원/병합 완료* (`{orig_name}`)\n"
                        f"• 파일 내 전체 기사: {result.get('fetched_count', 0)}건\n"
                        f"• 신규 반영: *{result.get('inserted_count', 0)}건*\n"
                        f"• 중복 건너뜀: {result.get('skipped_count', 0)}건"
                    )
                )
            elif status == "rejected":
                say(
                    text=(
                        f"🚫 *파일 반영 거절* (`{orig_name}`)\n"
                        f"• 사유: {result.get('message')}\n"
                        f"• 기존 저장소 데이터는 변경되지 않았어요."
                    )
                )
            else:
                say(text=f"⚠️ 복원 실패 (`{orig_name}`): {result.get('message')}")


def format_sync_result(data: dict, direction_label: str) -> str:
    if data.get("status") != "ok":
        return f"⚠️ 동기화 실패 ({direction_label}): {data.get('message', '알 수 없는 오류')}"

    limit_val = data.get("limit")
    limit_text = f"최근 {limit_val}건" if limit_val else "전체"
    return (
        f"🔄 *동기화 완료* ({direction_label})\n"
        f"• 대상 범위: {limit_text} (조회 {data.get('fetched_count', 0)}건)\n"
        f"• 신규 반영: *{data.get('inserted_count', 0)}건*\n"
        f"• 중복 건너뜀: {data.get('skipped_count', 0)}건"
    )


def handle_storage_action(
    tool: str,
    user_text: str,
    chosen: Dict[str, Any],
    event: Dict[str, Any],
    say: Any,
    client: Any,
    mcp_tools: Any,
) -> None:
    if tool == "set_storage_backend":
        target_mode = resolve_backend_value_with_llm(chosen.get("storage_backend"), user_text=user_text)
        if not target_mode:
            cur = json.loads(mcp_tools.get_storage_status(sqlite_path=DEFAULT_SQLITE_PATH)).get("active_backend")
            target_mode = "postgres" if cur == "sqlite" else "sqlite"
        data = json.loads(mcp_tools.set_storage_backend(target_mode))
        say(text=f"✅ 저장소 백엔드를 `{data.get('STORAGE_BACKEND')}`로 변경했어요.")
        return

    if tool == "export_sqlite_file":
        info = json.loads(mcp_tools.export_sqlite_file(sqlite_path=DEFAULT_SQLITE_PATH))
        if info.get("status") != "ok":
            say(text=f"❌ {info.get('message') or 'SQLite 파일을 찾지 못했어요.'}")
            return
        file_path = info["file_path"]
        filename = info.get("filename") or "news_store.sqlite"
        size_kb = round((info.get("size_bytes") or 0) / 1024, 1)
        try:
            client.files_upload_v2(
                channel=event.get("channel"),
                file=file_path,
                filename=filename,
                title=filename,
                initial_comment=f"📦 `{filename}` ({size_kb} KB) 파일을 추출했어요. 클릭해서 다운로드하세요.",
            )
        except Exception as e:
            say(text=f"❌ Slack 파일 업로드 실패: {e}")
        return

    if tool == "get_storage_status":
        data = json.loads(mcp_tools.get_storage_status(sqlite_path=DEFAULT_SQLITE_PATH))
        active_mode = (data.get("active_backend") or "미확인").strip().lower()
        mode_label = (
            f"`{active_mode}` (파일 저장중)"
            if active_mode == "sqlite"
            else f"`{active_mode}` (neon db로 저장중)"
            if active_mode == "postgres"
            else f"`{active_mode}`"
        )
        sq = data.get("sqlite") or {}
        pg = data.get("postgres") or {}
        sq_status = "정상 준비됨" if sq.get("exists") else "미준비"
        pg_status = (
            "정상 연결됨"
            if pg.get("connected")
            else "연결 실패"
            if pg.get("configured")
            else "미설정"
        )
        say(
            text=(
                f"📌 *저장소 상태*\n"
                f"• 현재 모드: {mode_label}\n"
                f"• SQLite: {sq_status}\n"
                f"• PostgreSQL: {pg_status}"
            )
        )
        return

    if tool == "get_storage_stats":
        raw_backend = resolve_backend_value_with_llm(chosen.get("target_backend"), user_text=user_text)
        explicit_keywords = ("sqlite", "파일", "로컬", "postgres", "postgresql", "neon", "네온")
        has_explicit_backend = any(k in (user_text or "").lower() for k in explicit_keywords)
        target_backend = raw_backend if has_explicit_backend else None
        data = json.loads(
            mcp_tools.get_storage_stats(
                target_backend=target_backend,
                sqlite_path=DEFAULT_SQLITE_PATH,
            )
        )
        backend = (data.get("backend") or "미확인").strip().lower()
        backend_label = (
            f"`{backend}` (파일 저장중)"
            if backend == "sqlite"
            else f"`{backend}` (neon db로 저장중)"
            if backend == "postgres"
            else f"`{backend}`"
        )
        if data.get("status") != "ok":
            say(
                text=(
                    f"📈 *저장소 통계*\n"
                    f"• 대상 저장소: {backend_label}\n"
                    f"• 상태: {data.get('message') or '통계를 불러올 수 없어요.'}"
                )
            )
            return

        by_source = data.get("by_source") or []
        by_category = data.get("by_category") or []
        src_text = ", ".join(f"{x['name']} {x['count']}건" for x in by_source) if by_source else "데이터 없음"
        cat_text = ", ".join(f"{x['name']} {x['count']}건" for x in by_category) if by_category else "데이터 없음"
        oldest = data.get("oldest_published_at") or "없음"
        latest = data.get("latest_published_at") or "없음"
        period_text = f"`{oldest}` ~ `{latest}`" if (oldest != "없음" and latest != "없음") else f"`{latest}`"
        say(
            text=(
                f"📈 *저장소 통계*\n"
                f"• 대상 저장소: {backend_label}\n"
                f"• 총 기사 수: *{data.get('total_count', 0)}건*\n"
                f"• 언론사별(상위): {src_text}\n"
                f"• 분야별(상위): {cat_text}\n"
                f"• 기사 기간: {period_text}"
            )
        )
        return

    if tool == "sync_sqlite_to_postgres":
        sync_limit = resolve_positive_int_with_llm(chosen.get("limit"), user_text=user_text, default=None)
        data = json.loads(
            mcp_tools.sync_sqlite_to_postgres(
                limit=sync_limit,
                sqlite_path=DEFAULT_SQLITE_PATH,
            )
        )
        say(text=format_sync_result(data, "SQLite → Neon DB"))
        return

    if tool == "sync_postgres_to_sqlite":
        sync_limit = resolve_positive_int_with_llm(chosen.get("limit"), user_text=user_text, default=None)
        data = json.loads(
            mcp_tools.sync_postgres_to_sqlite(
                limit=sync_limit,
                sqlite_path=DEFAULT_SQLITE_PATH,
            )
        )
        say(text=format_sync_result(data, "Neon DB → SQLite"))
        return

    if tool == "delete_news_records":
        merged = extract_delete_args_with_llm(user_text, chosen)
        raw_delete_all = merged.get("delete_all")
        delete_all_val = (
            raw_delete_all
            if isinstance(raw_delete_all, bool)
            else str(raw_delete_all or "").strip().lower() in ("true", "1", "yes")
        )
        target_backend = resolve_backend_value_with_llm(
            clean_str_arg(merged.get("target_backend")),
            user_text=user_text,
        )

        if delete_all_val:
            source_val = category_val = keyword_val = None
            days_val = limit_val = None
            order_val = "newest"
        else:
            source_val = clean_str_arg(merged.get("source"))
            category_val = clean_str_arg(merged.get("category"))
            keyword_val = clean_str_arg(merged.get("keyword"))
            days_val = resolve_positive_int_with_llm(merged.get("days"), user_text=user_text, default=None)
            limit_val = resolve_positive_int_with_llm(merged.get("limit"), user_text=user_text, default=None)
            raw_order = str(clean_str_arg(merged.get("order")) or "").strip().lower()
            order_val = "oldest" if raw_order in ("oldest", "asc") else "newest"

        data = json.loads(
            mcp_tools.delete_news_records(
                source=source_val,
                category=category_val,
                keyword=keyword_val,
                days=days_val,
                limit=limit_val,
                order=order_val,
                delete_all=delete_all_val,
                target_backend=target_backend,
                sqlite_path=DEFAULT_SQLITE_PATH,
            )
            or "{}"
        )
        if not isinstance(data, dict) or data.get("status") != "ok":
            say(text=f"⚠️ 삭제 처리 안내: {data.get('message') if isinstance(data, dict) else '삭제할 수 없었어요.'}")
            return

        cond_parts = []
        if (data.get("filters") or {}).get("delete_all"):
            cond_parts.append("전체 데이터")
        if source_val:
            cond_parts.append(f"언론사={source_val}")
        if category_val:
            cond_parts.append(f"분야={category_val}")
        if keyword_val:
            cond_parts.append(f"키워드={keyword_val}")
        if days_val:
            cond_parts.append(f"최근 {days_val}일")
        if limit_val:
            order_label = "오래된" if order_val == "oldest" else "최근"
            cond_parts.append(f"{order_label} {limit_val}건")

        say(
            text=(
                f"🗑️ *뉴스 삭제 완료* (`{data.get('backend') or '알 수 없음'}`)\n"
                f"• 적용 조건: {', '.join(cond_parts) if cond_parts else '지정 조건'}\n"
                f"• 삭제된 기사 수: *{int(data.get('deleted_count') or 0)}건*"
            )
        )