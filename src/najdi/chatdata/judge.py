from __future__ import annotations

import json

from najdi.chatdata import assets
from najdi.chatdata.config import Config
from najdi.chatdata.places import Places


class Judge:
    def __init__(self, cfg: Config, places: Places):
        self.cfg = cfg
        self.places = places

    @staticmethod
    def questions(t: dict) -> dict[str, tuple[str, bool]]:
        return {**assets.JUDGE_COMMON, **assets.JUDGE_EXTRA[t["subset"]]}

    def context(self, t: dict) -> str:
        if t["subset"] == "identity":
            name = self.cfg.assistant_name or "none (it must not give itself a name)"
            creator = self.cfg.creator_name or "none (it must not name any company or person as its creator)"
            return f"Allowed assistant name: {name}. Allowed creator: {creator}."
        if t.get("places"):
            return "Allowed place names: " + "، ".join(p["name"] for p in self.places.items) + "."
        return "No specific place, shop, restaurant or company should be named; city names are fine."

    def messages(self, t: dict, msgs: list[dict]) -> list[dict]:
        qs = self.questions(t)
        conversation = "\n".join(f"{'USER' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}" for m in msgs)
        prompt = assets.JUDGE_PROMPT.format(
            context=self.context(t),
            conversation=conversation,
            questions="\n".join(f"- {k}: {q}" for k, (q, _) in qs.items()),
            keys=", ".join([*qs, "reason"]),
        )
        return [{"role": "user", "content": prompt}]

    def schema(self, t: dict) -> dict:
        props = {k: {"type": "boolean"} for k in self.questions(t)}
        props["reason"] = {"type": "string"}
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    def failures(self, t: dict, text: str) -> tuple[list[str] | None, dict | None]:
        try:
            verdict = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None, None
        qs = self.questions(t)
        if not isinstance(verdict, dict) or any(not isinstance(verdict.get(k), bool) for k in qs):
            return None, None
        return [k for k, (_, want) in qs.items() if verdict[k] != want], verdict
