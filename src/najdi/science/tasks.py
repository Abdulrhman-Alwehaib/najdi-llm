from __future__ import annotations

import random

from najdi.llm.output import format_anchor, turns_schema
from najdi.science import assets
from najdi.science.config import Config


class TaskFactory:
    def __init__(self, cfg: Config, anchors: list[list[dict]]):
        self.cfg = cfg
        self.anchors = anchors
        self.examples = [(e["user"], e["assistant"]) for e in cfg.domain.examples]

    def _rng(self, task_id: str) -> random.Random:
        return random.Random(f"{self.cfg.seed}-{task_id}")

    def _common(self, rng: random.Random) -> tuple[int, str, str, list]:
        weights = self.cfg.user_turn_weights
        n_turns = rng.choices(list(weights), weights=list(weights.values()))[0]
        persona, gender = rng.choice(assets.PERSONAS)
        anchors = rng.sample(self.anchors, rng.randint(*self.cfg.anchors_per_prompt))
        return n_turns, persona, gender, anchors

    def science(self, passage: dict) -> dict:
        task_id = f"{self.cfg.id_prefix}-{passage['id']}"
        rng = self._rng(task_id)
        n_turns, persona, gender, anchors = self._common(rng)
        return dict(
            id=task_id,
            set="science",
            topic=passage["label"],
            n_turns=n_turns,
            persona=persona,
            gender=gender,
            followups=rng.sample(assets.SCIENCE_FOLLOWUPS, n_turns - 1),
            anchors=anchors,
            passage_id=passage["id"],
            source_title=passage["title"],
            source_url=passage["url"],
            passage=passage["passage"],
            example=rng.choice(self.examples),
        )

    def math(self, k: int) -> dict:
        task_id = f"math-{k:05d}"
        rng = self._rng(task_id)
        n_turns, persona, gender, anchors = self._common(rng)
        topic = rng.choices(list(assets.MATH_TOPICS), weights=[w for _, w in assets.MATH_TOPICS.values()])[0]
        return dict(
            id=task_id,
            set="math_logic",
            topic=topic,
            n_turns=n_turns,
            persona=persona,
            gender=gender,
            followups=rng.sample(assets.MATH_FOLLOWUPS, n_turns - 1),
            anchors=anchors,
            idea=rng.choice(assets.MATH_IDEAS[topic]),
            context=rng.choice(assets.MATH_CONTEXTS),
            difficulty=rng.choice(assets.DIFFICULTIES),
            example=rng.choice(assets.MATH_EXAMPLES),
        )

    def messages(self, t: dict) -> list[dict]:
        anchors = "\n\n".join(
            f"مثال {i + 1}:\n{format_anchor(a, self.cfg.anchor_max_messages)}" for i, a in enumerate(t["anchors"])
        )
        asker = t["persona"] + ("، وهي أنثى فالمساعد يكلمها بصيغة المؤنث" if t["gender"] == "f" else "")
        followups = "، بعدها ".join(t["followups"])
        example = f"السائل: {t['example'][0]}\nالمساعد: {t['example'][1]}"
        if t["set"] == "science":
            user = assets.SCIENCE_PROMPT.format(
                topic=self.cfg.domain.topics[t["topic"]],
                asker=asker,
                n=t["n_turns"],
                followups=followups,
                example=example,
                anchors=anchors,
                title=t["source_title"],
                passage=t["passage"],
                extra_rules=self.cfg.domain.extra_rules,
            )
        else:
            user = assets.MATH_PROMPT.format(
                topic=assets.MATH_TOPICS[t["topic"]][0],
                idea=t["idea"],
                context=t["context"],
                difficulty=t["difficulty"],
                asker=asker,
                n=t["n_turns"],
                followups=followups,
                example=example,
                anchors=anchors,
            )
        return [{"role": "system", "content": assets.SYSTEM_PROMPT}, {"role": "user", "content": user}]

    @staticmethod
    def schema(t: dict) -> dict:
        extra = {"final_answer": {"type": "string"}} if t["set"] == "math_logic" else None
        return turns_schema(t["n_turns"], extra)
