from __future__ import annotations

import json
from collections import Counter, defaultdict

from najdi.chatdata.config import Config
from najdi.chatdata.filters import FILTERS
from najdi.chatdata.places import Places
from najdi.chatdata.seeds import SITUATIONS, persona_key

DESCRIPTIONS = {
    "helpful": "Multi-turn everyday help (3 to 5 user turns): trips, food, family, studies, work, budget, cars, phones, "
    "healthy habits, occasions, weather, sports, home, travel, hobbies, friends and neighbours. Trip suggestions only "
    "name real places from the list below.",
    "identity": "Greetings and questions about the assistant itself, answered honestly: it is an AI assistant with no "
    "personal life. Some rows go on to a normal help request.",
}


def coverage(kept: list[dict]) -> list[str]:
    seeds = defaultdict(list)
    for r in kept:
        seeds[r["subset"]].append(json.loads(r["seed"]))
    lines = []
    if seeds["helpful"]:
        h = seeds["helpful"]
        pairs = {(s["situation"], persona_key(s["persona"])) for s in h}
        lines += [
            f"helpful: {len({s['situation'] for s in h})} of {len(SITUATIONS)} situations, {len(pairs)} distinct pairs",
            "  topics: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(s["topic"] for s in h).items())),
            "  user turns: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(s["turns"] for s in h).items())),
            "  follow-ups: "
            + ", ".join(f"{k} {v}" for k, v in Counter(f for s in h for f in s["followups"]).most_common()),
            "  trips with the places list: " + str(sum(bool(s.get("places")) for s in h)),
        ]
    for subset, items in seeds.items():
        people = Counter((s["persona"]["gender"], s["persona"]["reveals_gender"]) for s in items)
        ages = Counter(s["persona"]["age"] for s in items)
        lines.append(
            f"  {subset} users: "
            + ", ".join(f"{g} {'shows gender' if shown else 'no cue'} {c}" for (g, shown), c in sorted(people.items()))
            + " | ages "
            + ", ".join(f"{k} {v}" for k, v in sorted(ages.items()))
        )
    if seeds["identity"]:
        i = seeds["identity"]
        lines.append(
            "identity: "
            + ", ".join(f"{k} {v}" for k, v in Counter(s["topic"] for s in i).most_common())
            + f" | {len({s['greeting'] for s in i})} greetings, {len({q for s in i for q in s['questions']})} questions"
        )
    return lines


def filter_table(attempts: list[dict]) -> list[tuple[str, int, int]]:
    removed = Counter(r["reason"] for r in attempts if r["status"] == "dropped")
    table, left = [], len(attempts)
    for name in FILTERS:
        table.append((name, left, removed.get(name, 0)))
        left -= removed.get(name, 0)
    return table


def dataset_card(cfg: Config, kept: list[dict], attempts: list[dict], places: Places) -> str:
    counts = Counter(r["subset"] for r in kept)
    subsets = [s for s in cfg.targets if counts[s]]
    configs = "".join(
        f"- config_name: {s}\n  data_files:\n  - split: train\n    path: data/{s}.jsonl\n" for s in subsets
    )
    table = "\n".join(f"| `{s}` | {counts[s]} | {DESCRIPTIONS.get(s, '')} |" for s in subsets)
    filters = "\n".join(f"| `{name}` | {reached} | {k} |" for name, reached, k in filter_table(attempts))
    place_list = "\n".join(f"- [{p['name']}]({p['url']})" for p in places.items)
    cov = "\n".join(f"    {line}" for line in coverage(kept))
    return f"""---
language:
- ar
pretty_name: Saudi Najdi normal conversations
task_categories:
- text-generation
configs:
{configs}---
# Saudi Najdi normal conversations

Najdi Saudi conversations that teach a chat model to behave like a helpful AI assistant in casual chat.

| subset | rows | what |
|---|---|---|
{table}

Each row: `messages` (no system message), `id`, `subset`, `topic`, `seed` (JSON) and `generator`.

## How it was made

Generated with `{cfg.engine.model}` (vLLM, thinking `{cfg.thinking}`, structured JSON output) from unique seeds:
situation, adult user persona, writing style and follow-up types. Real Najdi dialogues from
[{cfg.voho_id}](https://huggingface.co/datasets/{cfg.voho_id}) were shown as examples of how people speak.

| filter | reached | removed |
|---|---|---|
{filters}

Seed coverage:

{cov}

## Places

Trip conversations only name these places. Their summaries come from Arabic Wikipedia (CC BY-SA 4.0).

{place_list}
"""
