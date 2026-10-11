from __future__ import annotations

import random
import re
from collections.abc import Callable

from najdi.chatdata import assets

SITUATIONS = [(topic, s) for topic, (_, sits) in assets.TOPICS.items() for s in sits]


def split_tags(situation: str) -> tuple[list[str], str]:
    return re.findall(r"\[(\w+)\]", situation), re.sub(r"\[\w+\]\s*", "", situation).strip()


def make_persona(rng: random.Random, tags=(), reveal_p: float = 0.5) -> dict:
    ages = ["20s"] if "student" in tags else ["40s", "50s", "60plus"] if "bigkids" in tags else list(assets.AGES)
    age, gender = rng.choice(ages), rng.choice("mf")
    k = "mf".index(gender)
    jobs = [j[k] for j in assets.JOBS[age] if j[k] and ("student" not in tags or j[k].startswith("طالب"))]
    families = [
        f
        for f in assets.FAMILY[age]
        if (f[0] == "kids" or not {"kids", "bigkids"} & set(tags))
        and (f[0] in ("married", "kids") or "married" not in tags)
    ]
    family, city = rng.choice(families), rng.choice(list(assets.CITIES))
    return {
        "age": age,
        "gender": gender,
        "reveals_gender": rng.random() < reveal_p,
        "job": rng.choice(jobs),
        "family": family[1 + k],
        "family_code": family[0],
        "mood": rng.choice(assets.MOODS)[k],
        "city": city,
        "region": assets.CITIES[city],
    }


def persona_key(p: dict) -> tuple:
    return tuple(p[k] for k in ("age", "gender", "reveals_gender", "job", "family_code", "mood", "city"))


def pick_followups(rng: random.Random, n: int) -> list[str]:
    kinds, out = [k for k in assets.FOLLOWUPS if k != "thanks"], []
    for j in range(n):
        if j == n - 1 and n >= 2 and rng.random() < 0.35:
            out.append("thanks")
        else:
            out.append(rng.choice([k for k in kinds if not out or k != out[-1]]))
    return out


KEYS: dict[str, Callable[[dict], tuple]] = {
    "helpful": lambda s: (s["situation"], persona_key(s["persona"])),
    "identity": lambda s: (s["topic"], s["greeting"], tuple(s["questions"]), s["situation"], persona_key(s["persona"])),
}


class SeedBuilder:
    def __init__(self, seed: int):
        self.seed = seed
        self.order = list(range(len(SITUATIONS)))
        random.Random(f"{seed}-situations").shuffle(self.order)

    def helpful_seed(self, i: int, rng: random.Random) -> dict:
        topic, raw = SITUATIONS[self.order[i % len(SITUATIONS)]]
        tags, situation = split_tags(raw)
        turns = rng.randint(*assets.USER_TURNS["helpful"])
        return {
            "id": f"helpful-{i:05d}",
            "subset": "helpful",
            "topic": topic,
            "situation": situation,
            "persona": make_persona(rng, tags),
            "style": rng.randrange(len(assets.STYLES)),
            "turns": turns,
            "followups": pick_followups(rng, turns - 1),
        }

    def identity_seed(self, i: int, rng: random.Random) -> dict:
        kinds = list(assets.IDENTITY_KINDS)
        kind = rng.choices(kinds, weights=[w for _, w in assets.IDENTITY_KINDS.values()])[0]
        _, raw = rng.choice(SITUATIONS)
        tags, situation = split_tags(raw)
        turns = rng.randint(*assets.USER_TURNS["identity"])
        if kind.endswith("_then_help"):
            turns = max(turns, 2)
        return {
            "id": f"identity-{i:05d}",
            "subset": "identity",
            "topic": kind,
            "greeting": rng.choice(assets.GREETINGS),
            "questions": rng.sample(assets.IDENTITY_QUESTIONS, 2),
            "situation": situation if kind.endswith("_then_help") else "",
            "persona": make_persona(rng, tags, reveal_p=0.4),
            "style": rng.randrange(len(assets.STYLES)),
            "turns": turns,
        }

    def build(self, subset: str, n: int) -> list[dict]:
        make = getattr(self, f"{subset}_seed")
        key = KEYS[subset]
        seeds, used = [], set()
        for i in range(n):
            rng = random.Random(f"{self.seed}-seed-{make.__name__}-{i}")
            for _ in range(100):
                s = make(i, rng)
                if key(s) not in used:
                    break
            used.add(key(s))
            seeds.append(s)
        if len({key(s) for s in seeds}) != len(seeds):
            raise RuntimeError(f"duplicate {subset} seeds")
        return seeds
