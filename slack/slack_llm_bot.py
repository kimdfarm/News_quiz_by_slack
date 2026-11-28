import importlib.util
import os
import re
import sys
from typing import Any

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_THIS_DIR)
_MCP_DIR = os.path.join(_BASE_DIR, "mcp")

for _p in (_BASE_DIR, _THIS_DIR, _MCP_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from crawler.news.plan_sources_from_request import get_rss_crawling_guide, parse_help_topic
from slack.llm_router import choose_mcp_tool
from slack.quiz_handler import (
    handle_quiz_db_manage,
    handle_quiz_generate,
    handle_quiz_ui_submit,
    try_handle_quiz_thread_reply,
    handle_quiz_essay_modal_open,
    handle_quiz_essay_modal_submit,
    
)
from slack.schedule_worker import (
    build_schedule_request_with_llm,
    handle_stop_command,
    parse_schedule_plan,
    run_once_rss_request,
    start_rss_schedule,
)
from slack.storage_file_ops import handle_storage_action, process_slack_files
from sqlite_store import init_sqlite_db
from storage_backend import DEFAULT_SQLITE_PATH, get_active_storage_backend
load_dotenv()


def _resolve_server_path() -> str:
    candidates = [
        os.path.join(_MCP_DIR, "news_mcp_server.py"),
        os.path.join(_BASE_DIR, "news_mcp_server.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


_spec = importlib.util.spec_from_file_location("news_mcp_server", _resolve_server_path())
mcp_tools: Any = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(mcp_tools)

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))


def _clean_mention_text(text: str) -> str:
    return re.sub(r"<@[A-Z0-9_]+>", "", text or "").strip()


@app.event("app_mention")
def handle_mention(event, say, client):
    user_text = _clean_mention_text(event.get("text", ""))
    thread_ts = event.get("thread_ts")
    if thread_ts and user_text:
        if try_handle_quiz_thread_reply(
            thread_ts=str(thread_ts),
            answer_text=user_text,
            user_id=event.get("user") or "unknown",
            say=say,
        ):
            return

    if user_text.strip().lower() == "!stop":
        handle_stop_command(say)
        return

    help_topic = parse_help_topic(user_text)
    if help_topic:
        say(text=get_rss_crawling_guide(help_topic))
        return

    attached_files = event.get("files") or []
    if attached_files:
        process_slack_files(attached_files, say)
        if not user_text:
            return

    if not user_text:
        say(text="명령을 입력해 주세요. (`--help`로 확인)")
        return

    chosen = choose_mcp_tool(user_text)
    tool = chosen.get("tool", "none")

    if tool == "none":
        say(text="그런 기능은 없어요.")
        return

    if tool == "run_rss_by_request":
        run_once_rss_request(request_text=user_text, say=say, mcp_tools=mcp_tools)
        return

    if tool == "schedule_rss_collection":
        try:
            plan = parse_schedule_plan(user_text, chosen)
            plan["request_text"] = build_schedule_request_with_llm(
                user_text=user_text,
                chosen=chosen,
                call_llm_fn=None,
            )
            start_rss_schedule(
                plan=plan,
                channel_id=event.get("channel"),
                client=client,
                say=say,
                mcp_tools=mcp_tools,
            )
        except Exception as e:
            say(text=f"⚠️ 주기 수집 예약 실패: {e}")
        return

    if tool in ("get_recent_news_for_quiz", "create_quiz_with_template"):
        handle_quiz_generate(
            event=event,
            user_text=user_text,
            chosen=chosen,
            say=say,
            mcp_tools=mcp_tools,
        )
        return

    if tool == "manage_quiz_project_db":
        handle_quiz_db_manage(event=event, chosen=chosen, say=say, client=client)
        return

    handle_storage_action(
        tool=tool,
        user_text=user_text,
        chosen=chosen,
        event=event,
        say=say,
        client=client,
        mcp_tools=mcp_tools,
    )

@app.action("quiz_choice_select")
def handle_quiz_radio_select(ack):
    ack()


@app.action("quiz_choice_submit")
def handle_quiz_submit_button(ack, body, client):
    ack()
    handle_quiz_ui_submit(body=body, client=client)
@app.action("quiz_essay_modal_open")
def handle_essay_modal_open_action(ack, body, client):
    ack()
    handle_quiz_essay_modal_open(body=body, client=client)


@app.view("quiz_essay_modal_submit")
def handle_essay_modal_submit_view(ack, body, client):
    ack()
    handle_quiz_essay_modal_submit(body=body, client=client)


@app.event("message")
def handle_message_events(event, say):
    if event.get("subtype") == "bot_message" or event.get("bot_id"):
        return

    thread_ts = event.get("thread_ts")
    cleaned_text = _clean_mention_text(event.get("text") or "")
    if thread_ts and cleaned_text:
        if try_handle_quiz_thread_reply(
            thread_ts=str(thread_ts),
            answer_text=cleaned_text,
            user_id=event.get("user") or "unknown",
            say=say,
        ):
            return

    raw_text = (event.get("text") or "").strip()
    if re.search(r"<@[A-Z0-9_]+>", raw_text):
        return

    if raw_text.lower() == "!stop":
        handle_stop_command(say)
        return

    attached_files = event.get("files") or []
    if attached_files:
        process_slack_files(attached_files, say)


if __name__ == "__main__":
    current_backend = get_active_storage_backend()
    if current_backend == "postgres" and os.getenv("DATABASE_URL"):
        from database import init_db

        init_db()
    elif current_backend == "sqlite":
        init_sqlite_db(DEFAULT_SQLITE_PATH)

    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()