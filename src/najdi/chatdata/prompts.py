from __future__ import annotations

import random

from najdi.chatdata import assets
from najdi.chatdata.config import Config
from najdi.chatdata.places import Places
from najdi.llm.output import format_anchor, turns_schema


def spouse(gender: str) -> str:
    return "زوجته" if gender == "m" else "زوجها"


class PromptBuilder:
    def __init__(self, cfg: Config, anchors: dict[str, list[list[dict]]], places: Places):
        self.cfg = cfg
        self.anchors = anchors
        self.places = places
        self.name_rule = assets.NAME_RULE.format(name=cfg.assistant_name) if cfg.assistant_name else assets.NO_NAME_RULE
        self.creator_rule = (
            assets.CREATOR_RULE.format(name=cfg.creator_name) if cfg.creator_name else assets.NO_CREATOR_RULE
        )

    @staticmethod
    def persona_text(p: dict, style: int) -> str:
        k = "mf".index(p["gender"])
        who = (
            f"المستخدم {'رجل' if k == 0 else 'امرأة'} {assets.AGES[p['age']][k]}، {p['job']}، {p['family']}، "
            f"من {p['city']}، {'مزاجه' if k == 0 else 'مزاجها'} الحين: {p['mood']}. {assets.STYLES[style][k]}."
        )
        if not p["reveals_gender"]:
            return (
                who + "\nكلام المستخدم ما يبين أبد إذا هو رجل أو امرأة (ما يوصف نفسه بكلمة مذكرة أو مؤنثة)، "
                "فالمساعد يخاطبه بصيغة المذكر أو بصيغة عامة."
            )
        if k == 1:
            return (
                who
                + "\nكلامها يبين إنها امرأة (مثل: أنا محتارة، أنا تعبانة)، فالمساعد يخاطبها بصيغة المؤنث بعد ما يبين."
            )
        return who + "\nكلامه يبين إنه رجل (مثل: أنا محتار، أنا تعبان)."

    def _anchor_text(self, rng: random.Random) -> str:
        picked = rng.sample(self.anchors["enterprise"], rng.randint(1, 2)) + rng.sample(self.anchors["everyday"], 1)
        return "\n\n".join(
            f"مثال {i + 1}:\n{format_anchor(a, self.cfg.anchor_max_messages)}" for i, a in enumerate(picked)
        )

    def task(self, seed: dict) -> dict:
        rng = random.Random(f"{self.cfg.seed}-{seed['id']}")
        t = {**seed, "anchors": self._anchor_text(rng)}
        if seed["subset"] == "helpful" and seed["topic"] in assets.PLACE_TOPICS:
            t["places"] = [p["name"] for p in self.places.for_persona(seed["persona"], rng)]
        return t

    def messages(self, t: dict, avoid: list[str] = ()) -> list[dict]:
        persona = self.persona_text(t["persona"], t["style"])
        gender = t["persona"]["gender"]
        anchors = assets.ANCHORS_BLOCK.format(anchors=t["anchors"])
        avoid_text = (
            "لا تبدأ أي رد من ردود المساعد بهالعبارات، لأنها انستخدمت كثير: "
            + "، ".join(f"«{a}»" for a in avoid)
            + "\n"
            if avoid
            else ""
        )
        n = t["turns"]
        if t["subset"] == "helpful":
            if t.get("places"):
                lines = "\n".join(f"  {name}: {self.places.summaries[name]}" for name in t["places"])
                places = assets.PLACES_RULE.format(places=lines)
            else:
                places = assets.NO_PLACES_RULE
            user = assets.HELPFUL_PROMPT.format(
                topic=assets.TOPICS[t["topic"]][0],
                situation=t["situation"].replace("{spouse}", spouse(gender)),
                persona=persona,
                n=n,
                followups="، بعدها ".join(assets.FOLLOWUPS[f] for f in t["followups"]),
                places=places,
                anchors=anchors,
                avoid=avoid_text,
            )
        else:
            user = assets.IDENTITY_PROMPT.format(
                kind=assets.IDENTITY_KINDS[t["topic"]][0],
                persona=persona,
                n=n,
                plan=self._identity_plan(t, gender),
                name_rule=self.name_rule,
                creator_rule=self.creator_rule,
                anchors=anchors,
                avoid=avoid_text,
            )
        return [{"role": "system", "content": assets.SYSTEM_PROMPT}, {"role": "user", "content": user}]

    @staticmethod
    def _identity_plan(t: dict, gender: str) -> str:
        q1, q2 = t["questions"]
        kind = t["topic"]
        if kind.startswith("greeting"):
            first = f"- أول رسالة: تحية، مثل «{t['greeting']}» أو شي قريب منها."
        else:
            first = f"- أول رسالة: سؤال عن المساعد نفسه، مثل «{q1}» أو شي قريب منه."
        if kind.endswith("_then_help"):
            situation = t["situation"].replace("{spouse}", spouse(gender))
            rest = f"- الرسالة الثانية: يطلب مساعدة في موقف من حياته: {situation}. المساعد يساعده بجواب محدد وعملي."
            if t["turns"] > 2:
                rest += "\n- الرسالة الثالثة: متابعة طبيعية للطلب."
        elif kind == "identity":
            rest = f"- إذا فيه رسايل بعدها: يسأل سؤال ثاني عن المساعد، مثل «{q2}»، أو يعلق على جوابه."
        else:
            rest = "- إذا فيه رسايل بعدها: يرد بكلام قصير عادي (مثل: بخير الحمدلله وانت؟)، أو يسأل المساعد عن نفسه، أو يشكر."
        return first + "\n" + rest

    @staticmethod
    def schema(t: dict) -> dict:
        return turns_schema(t["turns"])
