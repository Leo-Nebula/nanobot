"""Structured trace logging for end-to-end agent debugging.

Each trace_event() call produces exactly ONE JSONL record (not multiple).
A ``turn_id`` field groups all events that belong to the same user-message
processing cycle, making it trivial to grep/filter a single turn.

Enable with:  NANOBOT_TRACE_LOG=1
"""

from __future__ import annotations

import json
import inspect
import os
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_logs_dir
from nanobot.utils.helpers import ensure_dir, safe_filename

_TRACE_ENABLED_VALUES = {"1", "true", "yes", "on", "debug"}
_TRACE_FILE_NAME = "agent_trace.jsonl"
_TRACE_SESSION_DIR = "trace_sessions"
_TRACE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("nanobot_trace_context", default={})
_WRITE_LOCK = threading.Lock()
_REDACTED = "<redacted>"
_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "id_token",
    "session_token",
    "auth_token",
    "bearer_token",
    "secret",
    "password",
    "cookie",
)
_NON_SENSITIVE_TOKEN_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "audio_tokens",
    "cached_tokens",
    "text_tokens",
    "image_tokens",
}


def new_turn_id() -> str:
    """Generate a short unique turn identifier (8 hex chars)."""
    return uuid.uuid4().hex[:8]


def trace_enabled() -> bool:
    value = os.getenv("NANOBOT_TRACE_LOG", "")
    return value.strip().lower() in _TRACE_ENABLED_VALUES


def _max_text_chars() -> int:
    raw = os.getenv("NANOBOT_TRACE_MAX_CHARS", "4000")
    try:
        return max(200, int(raw))
    except ValueError:
        return 4000


def _truncate(text: str, limit: int | None = None) -> str:
    max_chars = limit or _max_text_chars()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"


def _shorten(text: str, limit: int = 180) -> str:
    return _truncate(text.replace("\n", "\\n"), limit)


def _to_inline_text(value: Any, limit: int = 260) -> str:
    if isinstance(value, str):
        return _shorten(value, limit)
    try:
        return _shorten(json.dumps(value, ensure_ascii=False), limit)
    except TypeError:
        return _shorten(str(value), limit)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _NON_SENSITIVE_TOKEN_KEYS:
        return False
    if lowered == "token":
        return True
    return any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS)


def summarize_value(value: Any, *, _depth: int = 0) -> Any:
    if _depth >= 6:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, list):
        return [summarize_value(item, _depth=_depth + 1) for item in value]
    if isinstance(value, tuple):
        return [summarize_value(item, _depth=_depth + 1) for item in value]
    if isinstance(value, dict):
        summarized: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key == "_meta":
                continue
            if isinstance(key, str) and _is_sensitive_key(key):
                summarized[str(key)] = _REDACTED
                continue
            if (
                key == "url"
                and isinstance(item, str)
                and item.startswith("data:image/")
            ):
                summarized[str(key)] = "data:image/<omitted>"
                continue
            summarized[str(key)] = summarize_value(item, _depth=_depth + 1)
        return summarized
    if hasattr(value, "model_dump"):
        return summarize_value(value.model_dump(), _depth=_depth + 1)
    if hasattr(value, "dict"):
        return summarize_value(value.dict(), _depth=_depth + 1)
    if hasattr(value, "__dict__"):
        return summarize_value(vars(value), _depth=_depth + 1)
    return _truncate(repr(value))


def summarize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        entry: dict[str, Any] = {
            "index": index,
            "role": message.get("role"),
        }
        if "name" in message:
            entry["name"] = message.get("name")
        if "tool_call_id" in message:
            entry["tool_call_id"] = message.get("tool_call_id")
        if "tool_calls" in message:
            entry["tool_calls"] = summarize_value(message.get("tool_calls"))
        if "reasoning_content" in message:
            entry["reasoning_content"] = summarize_value(message.get("reasoning_content"))
        if "thinking_blocks" in message:
            entry["thinking_blocks"] = summarize_value(message.get("thinking_blocks"))
        entry["content"] = summarize_value(message.get("content"))
        summarized.append(entry)
    return summarized


def _compact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        item: dict[str, Any] = {"role": role}
        content = message.get("content")
        if isinstance(content, str):
            item["content"] = _shorten(content)
            item["chars"] = len(content)
        elif isinstance(content, list):
            item["content"] = _shorten(json.dumps(summarize_value(content), ensure_ascii=False), 220)
            item["blocks"] = len(content)
        else:
            item["content"] = summarize_value(content)
        if message.get("tool_calls"):
            item["tool_calls"] = [
                tc.get("function", {}).get("name") if isinstance(tc, dict) else str(tc)
                for tc in message.get("tool_calls", [])
            ]
        compact.append(item)
    return compact


def _compact_summary(event: str, context: dict[str, Any], payload: dict[str, Any]) -> str:
    if event == "turn_start":
        return (
            f"▶ TURN {payload.get('turn_id')} channel={payload.get('channel')} "
            f"sender={payload.get('sender_id')} type={payload.get('message_type')} "
            f"content={_shorten(str(payload.get('content', '')))}"
        )
    if event == "turn_end":
        tools = payload.get("tools_used") or []
        return (
            f"◀ TURN {payload.get('turn_id')} tools=[{', '.join(tools)}] "
            f"content={_shorten(str(payload.get('content') or payload.get('final_content') or ''))}"
        )
    if event == "feishu_raw_inbound":
        return (
            f"feishu raw message_id={payload.get('message_id')} "
            f"type={payload.get('message_type')} chat={payload.get('chat_id')} "
            f"raw={_shorten(str(payload.get('raw_content', '')))}"
        )
    if event == "channel_inbound":
        return (
            f"channel inbound session={payload.get('session_key')} "
            f"content={_shorten(str(payload.get('content', '')))}"
        )
    if event == "session_state":
        session = payload.get("session") or {}
        return (
            f"session={context.get('session_key')} history={session.get('history_count')} "
            f"stored={session.get('message_count')} last_consolidated={session.get('last_consolidated')} "
            f"memory_chars={len(str(payload.get('long_term_memory') or ''))}"
        )
    if event == "llm_input_prepared":
        messages = payload.get("messages") or []
        return f"prepared messages={len(messages)} summary={_shorten(json.dumps(_compact_messages(messages), ensure_ascii=False), 320)}"
    if event == "loop_round_start":
        return (
            f"round={context.get('iteration')} messages={payload.get('message_count')} "
            f"tools={payload.get('tool_count')}"
        )
    if event == "llm_request":
        body = (payload.get("payload") or {}) if isinstance(payload.get("payload"), dict) else {}
        messages = body.get("messages") or []
        return (
            f"request round={context.get('iteration')} model={body.get('model')} "
            f"messages={len(messages)} tools={len(body.get('tools') or [])} "
            f"summary={_shorten(json.dumps(_compact_messages(messages[-4:]), ensure_ascii=False), 320)}"
        )
    if event in {"llm_raw_response", "loop_round_response"}:
        if event == "llm_raw_response":
            response = payload.get("response") or {}
            choices = response.get("choices") or []
            first = choices[0] if choices else {}
            usage = response.get("usage") or {}
            return (
                f"raw response round={context.get('iteration')} finish={first.get('finish_reason')} "
                f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
                f"content={_shorten(str(first.get('content') or ''))}"
            )
        return (
            f"round={context.get('iteration')} finish={payload.get('finish_reason')} "
            f"prompt={((payload.get('usage') or {}).get('prompt_tokens'))} "
            f"completion={((payload.get('usage') or {}).get('completion_tokens'))} "
            f"tool_calls={len(payload.get('tool_calls') or [])} "
            f"content={_shorten(str(payload.get('content') or ''))}"
        )
    if event == "tool_result":
        return (
            f"round={context.get('iteration')} tool={payload.get('tool_name')} "
            f"args={_shorten(json.dumps(payload.get('arguments'), ensure_ascii=False), 200)} "
            f"result={_shorten(json.dumps(payload.get('result'), ensure_ascii=False), 220)}"
        )
    if event == "loop_final_output":
        return f"session={context.get('session_key')} final={_shorten(str(payload.get('final_content') or ''), 260)}"
    if event == "outbound_message":
        return f"outbound channel={payload.get('channel')} chat={payload.get('chat_id')} content={_shorten(str(payload.get('content') or ''), 220)}"
    return _shorten(json.dumps(payload, ensure_ascii=False), 240)


def _caller_info() -> dict[str, Any]:
    frame = inspect.currentframe()
    try:
        current_file = Path(__file__).resolve()
        while frame is not None:
            frame = frame.f_back
            if frame is None:
                break
            code = frame.f_code
            file_path = Path(code.co_filename).resolve()
            if file_path == current_file:
                continue
            module = frame.f_globals.get("__name__", "unknown")
            return {
                "module": module,
                "function": code.co_name,
                "line": frame.f_lineno,
                "file": str(file_path),
            }
    finally:
        del frame
    return {
        "module": "unknown",
        "function": "unknown",
        "line": 0,
        "file": "unknown",
    }


def _source_label(source: dict[str, Any]) -> str:
    return f"{source.get('module')}:{source.get('function')}:{source.get('line')}"


def _find_session_key(context: dict[str, Any], payload: dict[str, Any]) -> str | None:
    direct = context.get("session_key") or payload.get("session_key")
    if isinstance(direct, str) and direct:
        return direct

    inbound = payload.get("inbound_message")
    if isinstance(inbound, dict):
        session_key = inbound.get("session_key")
        if isinstance(session_key, str) and session_key:
            return session_key

    return None


def _session_trace_path(logs_dir: Path, session_key: str | None) -> Path | None:
    if not session_key:
        return None
    session_dir = ensure_dir(logs_dir / _TRACE_SESSION_DIR)
    safe_key = safe_filename(session_key.replace(":", "__")) or "unknown_session"
    return session_dir / f"{safe_key}.jsonl"


def _build_record(
    ts: str,
    event: str,
    source: dict[str, Any],
    context: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build a single flat JSONL record for one trace event."""
    session_key = _find_session_key(context, payload)
    record: dict[str, Any] = {
        "ts": ts,
        "event": event,
        "session_key": session_key,
        "summary": _compact_summary(event, context, payload),
    }
    if context.get("turn_id"):
        record["turn_id"] = context["turn_id"]
    if context.get("iteration"):
        record["iteration"] = context["iteration"]
    record["data"] = payload
    record["source"] = _source_label(source)
    return record


@contextmanager
def trace_scope(**fields: Any):
    current = dict(_TRACE_CONTEXT.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    token = _TRACE_CONTEXT.set(current)
    try:
        yield current
    finally:
        _TRACE_CONTEXT.reset(token)


def trace_event(event: str, **payload: Any) -> None:
    if not trace_enabled():
        return

    source = summarize_value(_caller_info())
    context = summarize_value(_TRACE_CONTEXT.get())
    payload = summarize_value(payload)
    ts = datetime.now().isoformat()

    logger.debug("[trace:{}] {} | {}", event, _source_label(source), _compact_summary(event, context, payload))

    logs_dir = get_logs_dir()
    path = logs_dir / _TRACE_FILE_NAME
    record = _build_record(ts, event, source, context, payload)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    session_path = _session_trace_path(logs_dir, record.get("session_key"))
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
        if session_path is not None:
            with open(session_path, "a", encoding="utf-8") as handle:
                handle.write(line)
