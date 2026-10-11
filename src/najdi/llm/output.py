from __future__ import annotations

import json
import re


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if text.lstrip().startswith("{"):
        return text.strip()
    return ""


def roles_ok(msgs: list[dict]) -> bool:
    return len(msgs) % 2 == 0 and all(m["role"] == ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(msgs))


def parse_turns(text: str, turns: int | tuple[int, int]) -> tuple[list[dict], dict] | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    items = obj.get("turns") if isinstance(obj, dict) else None
    lo, hi = (turns, turns) if isinstance(turns, int) else turns
    if not isinstance(items, list) or not lo <= len(items) <= hi:
        return None
    msgs = []
    for turn in items:
        if not isinstance(turn, dict):
            return None
        for role in ("user", "assistant"):
            content = turn.get(role)
            if not isinstance(content, str) or not content.strip() or "<think>" in content or "</think>" in content:
                return None
            msgs.append({"role": role, "content": re.sub(r"\n{3,}", "\n\n", content.strip())})
    return (msgs, obj) if roles_ok(msgs) else None


def turns_schema(n: int, extra: dict | None = None) -> dict:
    turn = {
        "type": "object",
        "properties": {"user": {"type": "string"}, "assistant": {"type": "string"}},
        "required": ["user", "assistant"],
        "additionalProperties": False,
    }
    props = {"turns": {"type": "array", "items": turn, "minItems": n, "maxItems": n}, **(extra or {})}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def format_anchor(msgs: list[dict], max_messages: int) -> str:
    return "\n".join(f"{'أ' if m['role'] == 'user' else 'ب'}: {m['content'].strip()}" for m in msgs[:max_messages])
