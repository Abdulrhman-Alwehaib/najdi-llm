from __future__ import annotations

import json
from collections import Counter

from najdi.science.config import Config

FILTERS = [
    "format",
    "script",
    "markdown",
    "length",
    "source_mention",
    "sensitive",
    "najdi_words",
    "dialect_clf",
    "dedup",
    "math_check",
]


def rules(cfg: Config) -> dict[str, str]:
    f = cfg.filters
    return {
        "format": "valid JSON with 2-4 user turns; roles alternate, starting with user and ending with assistant",
        "script": f"no CJK or other non-Arabic scripts; no run of more than {f.max_english_run} English words",
        "markdown": "plain text only: no bold, headings, bullets, tables or LaTeX",
        "length": f"assistant replies {f.assist_words[0]}-{f.assist_words[1]} words; user turns "
        f"{f.user_words[0]}-{f.user_words[1]} words",
        "source_mention": "never mentions Wikipedia or the article",
        "sensitive": "no sexual topics; medical topics are allowed",
        "najdi_words": f"every assistant reply has at least {f.min_saudi_per_100w:g} Saudi marker words per 100 "
        f"Arabic words and at least {f.min_saudi_per_reply}; Fusha markers at most {f.max_fusha_ratio:.0%} of "
        "the Saudi ones in every message",
        "dialect_clf": f"`{f.dialect_classifier}` labels every assistant reply as Gulf",
        "dedup": f"opening question is not a near-duplicate (character 4-gram Jaccard < {f.dedup_jaccard}); "
        "each passage is used once",
        "math_check": "math/logic: solved again independently; kept only if both final answers match",
    }


def dataset_card(cfg: Config, kept: list[dict], attempts: list[dict]) -> str:
    n = len(kept)
    size = "n<1K" if n < 1000 else "1K<n<10K" if n < 10000 else "10K<n<100K"
    removed = Counter(a["reason"] for a in attempts if a["status"] == "dropped")
    topics = "\n".join(
        f"| {s} | {t} | {c} |" for (s, t), c in sorted(Counter((r["set"], r["topic"]) for r in kept).items())
    )
    filters = "\n".join(f"| {name} | {rule} | {removed.get(name, 0)} |" for name, rule in rules(cfg).items())
    tags = "".join(f"\n- {t}" for t in ["saudi", "najdi", "arabic-dialect", "synthetic", *cfg.domain.tags])
    example = json.dumps(kept[0], ensure_ascii=False, indent=2) if kept else "{}"
    return f"""---
language:
- ar
license: cc-by-sa-4.0
task_categories:
- text-generation
tags:{tags}
size_categories:
- {size}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# {cfg.domain.title}

{n} multi-turn conversations in Najdi Saudi Arabic. {cfg.domain.description}

Each conversation has 2 to 4 user turns and no system message.

## How it was made

- Generator: `{cfg.engine.model}` with vLLM, thinking `{cfg.thinking}`; thinking text is removed.
- Style examples: 2 to 3 `everyday` dialogues from [{cfg.voho_id}](https://huggingface.co/datasets/{cfg.voho_id}) per prompt.
- Grounding: each topic conversation is based on the lead of one Arabic Wikipedia article
  ([{cfg.wiki.dataset}](https://huggingface.co/datasets/{cfg.wiki.dataset}), `{cfg.wiki.subset}`), shortlisted by keywords and
  labelled by the generator. `source_title` and `source_url` name the article.
- Math and logic problems are synthetic and kept only when an independent second solution agrees.

## Topics

| set | topic | rows |
|---|---|---|
{topics}

## Filters

| filter | rule | rows removed |
|---|---|---|
{filters}

## Format

```json
{example}
```

## License

Rows derived from Arabic Wikipedia are CC BY-SA 4.0, so the dataset is released under CC BY-SA 4.0.
"""
