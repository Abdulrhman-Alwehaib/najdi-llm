from __future__ import annotations

import json
from collections import Counter

from najdi import hub
from najdi.mix.config import CLEAN_STEPS, SOURCES, Config, Report

LICENSES = {
    "cidar": ("arbml/CIDAR", "CC BY-NC 4.0 (translated from Alpaca-derived data)"),
    "aya": ("CohereLabs/aya_dataset, Standard Arabic rows", "Apache 2.0"),
}


def composition(rows: list[dict]):
    import pandas as pd

    df = pd.DataFrame({"source": [r["source"] for r in rows], "tokens": [r["_tokens"] for r in rows]})
    g = df.groupby("source")["tokens"].agg(
        rows="size", tokens="sum", median="median", p95=lambda s: s.quantile(0.95), max="max"
    )
    g = g.reindex([s for s in SOURCES if s in g.index])
    g["row_share"] = g["rows"] / g["rows"].sum()
    g["token_share"] = g["tokens"] / g["tokens"].sum()
    g.loc["total"] = [
        g["rows"].sum(),
        g["tokens"].sum(),
        df["tokens"].median(),
        df["tokens"].quantile(0.95),
        df["tokens"].max(),
        1.0,
        1.0,
    ]
    return g.astype({"rows": int, "tokens": int, "max": int}).round(
        {"median": 0, "p95": 0, "row_share": 3, "token_share": 3}
    )


def cleaning_table(report: Report):
    import pandas as pd

    df = pd.DataFrame({step: report.removed[step] for step in CLEAN_STEPS}).T.reindex(
        index=CLEAN_STEPS, columns=SOURCES
    )
    df = df.fillna(0).astype(int)
    df.loc["loaded"] = [report.loaded[s] for s in SOURCES]
    df = df.loc[["loaded", *CLEAN_STEPS]]
    df["total"] = df.sum(axis=1)
    return df


def _md_table(df, shares=("row_share", "token_share")) -> str:
    cols = list(df.columns)
    lines = ["| source | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [f"{row[c]:.1%}" if c in shares else f"{int(row[c]):,}" for c in cols]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dataset_card(cfg: Config, repo: str, splits: dict, unique: dict, report: Report, train_composition) -> str:
    fusha_name, fusha_license = LICENSES[cfg.fusha_source]
    n = sum(len(p) for p in splits.values())
    per_split = {name: Counter(r["source"] for r in part) for name, part in splits.items()}
    split_table = "\n".join(
        ["| source | unique rows | train | validation | test |", "|---|---|---|---|---|"]
        + [
            f"| {s} | {unique[s]:,} | {per_split['train'][s]:,} | {per_split['validation'][s]:,} | {per_split['test'][s]:,} |"
            for s in SOURCES
        ]
        + [
            f"| total | {sum(unique.values()):,} | {len(splits['train']):,} | {len(splits['validation']):,} "
            f"| {len(splits['test']):,} |"
        ]
    )
    clean = cleaning_table(report)
    clean_table = "\n".join(
        ["| step | " + " | ".join(SOURCES) + " | total |", "|---|" + "---|" * (len(SOURCES) + 1)]
        + [f"| {step} | " + " | ".join(str(v) for v in row) + " |" for step, row in clean.iterrows()]
    )
    train_table = _md_table(train_composition[["rows", "row_share", "tokens", "token_share", "median", "p95", "max"]])
    system_msg = json.dumps({"role": "system", "content": cfg.fusha_system_prompt}, ensure_ascii=False)
    abshir = ", ".join(f"{s} {c} of {k}" for s, (k, c) in report.abshir.items())
    nc = (
        "\n**Non-commercial research use only.** The mix includes CIDAR, treated as CC BY-NC 4.0.\n"
        if cfg.fusha_source == "cidar"
        else ""
    )
    return f"""---
language:
- ar
license: other
license_name: mixed
task_categories:
- text-generation
tags:
- saudi
- najdi
- arabic-dialect
- sft
size_categories:
- {"1K<n<10K" if n < 10000 else "10K<n<100K"}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

# Najdi Saudi chat training mix
{nc}
Saudi rows have no system message; Fusha rows start with `{system_msg}`.

## Splits

Test is drawn first and never used for training. Validation is {cfg.val_frac:.0%} of the rest per source. Every
helpful train row appears {cfg.helpful_repeat} times (copies end in `-copy2`).

{split_table}

## Train composition (`{cfg.tokenizer}` tokens)

{train_table}

## Sources

| source | from | license |
|---|---|---|
| `helpful`, `identity` | `{hub.resolve(cfg.chat_repo)}` | CC BY-SA 4.0 |
| `science`, `math_logic` | {", ".join(f"`{hub.resolve(r)}`" for r in cfg.science_repos)} | CC BY-SA 4.0 |
| `voho_everyday`, `voho_enterprise` | `{hub.resolve(cfg.voho_id)}` | Apache 2.0 |
| `fusha` | {fusha_name} | {fusha_license} |

## Cleaning

{clean_table}

In {cfg.abshir_cut_frac:.0%} of the helpful and identity replies that opened with "أبشر" ({abshir}) that word was removed.

```python
from datasets import load_dataset
mix = load_dataset("{repo}", token=True)
```
"""
