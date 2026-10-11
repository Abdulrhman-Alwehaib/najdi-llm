from __future__ import annotations

import logging
from pathlib import Path

from najdi import hub, jsonl
from najdi.env import hf_token
from najdi.mix.card import cleaning_table, composition, dataset_card
from najdi.mix.config import SOURCES, Config, Report, counts_line
from najdi.mix.sources import Cleaner, chat_text, load
from najdi.mix.splits import Splitter
from najdi.notify import telegram
from najdi.text.dedup import MinHashIndex

log = logging.getLogger(__name__)

PRIORITY = {
    s: i
    for i, s in enumerate(["helpful", "identity", "science", "math_logic", "fusha", "voho_everyday", "voho_enterprise"])
}


def dedup(rows: list[dict], cfg: Config, report: Report) -> list[dict]:
    index = MinHashIndex(cfg.near_dup_jaccard, seed=cfg.seed)
    kept = []
    for r in sorted(rows, key=lambda r: PRIORITY[r["source"]]):
        found = index.add(chat_text(r))
        if found:
            report.removed[f"{found}_duplicate"][r["source"]] += 1
        else:
            kept.append(r)
    return kept


def count_tokens(rows: list[dict], cfg: Config) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer, token=hf_token())
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    for r, ids in zip(rows, tok(texts, add_special_tokens=False)["input_ids"]):
        r["_tokens"] = len(ids)


def run(cfg: Config) -> None:
    report = Report()
    telegram(f"mix started: {cfg.repo_id}")
    try:
        rows = load(cfg, report)
        log.info("loaded: %s", counts_line(rows))
        rows = Cleaner(cfg, report)(rows)
        log.info("after cleaning: %s | abshir cut: %s", counts_line(rows), report.abshir)
        rows = dedup(rows, cfg, report)
        count_tokens(rows, cfg)
        rows = report.drop_where(rows, "too_long", lambda r: r["_tokens"] > cfg.max_seq_len)
        log.info("clean pool: %s", counts_line(rows))
        splits, unique = Splitter(cfg)(rows)
        tables = {name: composition(part) for name, part in splits.items()}
        for name, table in tables.items():
            log.info("%s composition\n%s", name, table.to_string())
        log.info("cleaning\n%s", cleaning_table(report).to_string())

        out = Path(cfg.output_dir)
        files = {}
        for name, part in splits.items():
            path = jsonl.write(
                out / "data" / f"{name}.jsonl", ({k: r[k] for k in ("id", "source", "messages")} for r in part)
            )
            files[f"data/{name}.jsonl"] = path
        repo = hub.resolve(cfg.repo_id) if cfg.push else cfg.repo_id
        card = dataset_card(cfg, repo, splits, unique, report, tables["train"])
        (out / "README.md").write_text(card, encoding="utf-8")
        files["README.md"] = out / "README.md"
        if cfg.push:
            repo_id = hub.push(cfg.repo_id, files, f"training mix: {counts_line(splits['train'])}")
            log.info("pushed to https://huggingface.co/datasets/%s", repo_id)
    except Exception as exc:
        telegram(f"mix crashed: {type(exc).__name__}: {exc}")
        raise
    telegram("mix done: " + ", ".join(f"{name} {len(part)}" for name, part in splits.items()))
    log.info("unique rows: %s", {s: unique[s] for s in SOURCES})
