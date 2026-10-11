from __future__ import annotations

import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from najdi import hub, jsonl
from najdi.text.arabic import WORD_END, alternation, loose, normalize, words
from najdi.text.openers import PRAISE_NEARBY, cut_opener, filler_pattern, opener_counts
from najdi.voho import load_dialogues

log = logging.getLogger(__name__)

_SKY = re.compile(loose("السماء") + "|" + loose("السما") + WORD_END)
_BLUE = re.compile(alternation(["أزرق", "زرقاء", "زرقا", "الزرقة", "لونها", "لون السماء"]))


@dataclass
class Config:
    science_repos: list[str] = field(default_factory=list)
    voho_id: str = "VohoAI/voho-saudi-dialogues"
    voho_clean_repo: str = ""
    extra_filler_openers: list[str] = field(default_factory=list)
    voho_flirt_terms: list[str] = field(default_factory=list)
    push: bool = True
    seed: int = 1234
    output_dir: str = "runs/clean"


class FlirtRemover:
    def __init__(self, terms: list[str]):
        self.pattern = re.compile(
            r"(?P<pre>[ \t]*[،,]?[ \t]*)(?<![^\W\d_])(?P<term>"
            + alternation(terms)
            + ")"
            + WORD_END
            + r"(?P<post>[ \t]*[،,]?[ \t]*)"
        )

    @staticmethod
    def _replacement(m: re.Match) -> str:
        before, after = m.string[: m.start()], m.string[m.end() :]
        if not before.strip() or before.rstrip(" \t").endswith("\n"):
            return ""
        if after.lstrip(" \t")[:1] in ("", ".", "!", "؟", "?", "…", "\n"):
            return ""
        around = m.group("pre") + m.group("post")
        return "، " if "،" in around or "," in around else " "

    def __call__(self, text: str) -> tuple[str, list[str]]:
        found = [" ".join(normalize(m.group("term")).split()) for m in self.pattern.finditer(text)]
        if not found:
            return text, found
        out = re.sub(r"(^|\n)[ \t]*[،,][ \t]*", r"\1", self.pattern.sub(self._replacement, text))
        return out.strip(), found


def clean_science(cfg: Config, out_dir: Path) -> None:
    extra = filler_pattern(cfg.extra_filler_openers)
    for name in cfg.science_repos:
        rows = jsonl.read(hub.download(name, "data/train.jsonl"))
        before = [m["content"] for r in rows for m in r["messages"] if m["role"] == "assistant"]
        cleaned, cut = [], 0
        for r in rows:
            msgs = []
            for m in r["messages"]:
                if m["role"] == "assistant":
                    text, changed = cut_opener(m["content"], extra)
                    cut += changed
                    m = {**m, "content": text}
                msgs.append(m)
            cleaned.append({**r, "messages": msgs})
        after = [m["content"] for r in cleaned for m in r["messages"] if m["role"] == "assistant"]
        left = [t for t in after if PRAISE_NEARBY.match(t)]
        log.info(
            "%s: %d rows, %d replies, %d openers cut, %d praise openers left",
            name,
            len(rows),
            len(before),
            cut,
            len(left),
        )
        for label, replies in (("before", before), ("after", after)):
            log.info("top openers %s: %s", label, opener_counts(replies, 20))
        sky = [
            r
            for r in rows
            if any(_SKY.search(m["content"]) for m in r["messages"])
            and any(_BLUE.search(m["content"]) for m in r["messages"])
        ]
        log.info("rows about the colour of the sky: %d", len(sky))
        path = jsonl.write(out_dir / f"{name}-clean.jsonl", cleaned)
        if cfg.push:
            card = (
                f"---\nlanguage:\n- ar\npretty_name: {name} (clean)\n---\n# {name} (clean)\n\n"
                f"Copy of `{hub.resolve(name)}` with the opening praise of the question cut from {cut} of "
                f"{len(before)} assistant replies. Everything else is unchanged.\n"
            )
            repo_id = hub.push(f"{name}-clean", {"data/train.jsonl": path, "README.md": card}, f"clean copy of {name}")
            log.info("pushed %s", repo_id)


def clean_voho(cfg: Config, out_dir: Path) -> None:
    remove = FlirtRemover(cfg.voho_flirt_terms)
    rows = load_dialogues(cfg.voho_id)
    cleaned, dropped, examples = [], 0, []
    terms, roles, changed_msgs, changed_rows = Counter(), Counter(), 0, 0
    for r in rows:
        msgs, hit, empty = [], False, False
        for m in r["messages"]:
            text, found = remove(m["content"])
            if found:
                hit, changed_msgs = True, changed_msgs + 1
                empty |= not words(text)
                terms.update(found)
                roles[m["role"]] += 1
                examples.append((m["content"], text))
            msgs.append({**m, "content": text})
        if empty:
            dropped += 1
            continue
        changed_rows += hit
        cleaned.append({**r, "messages": msgs})
    log.info(
        "%s: %d dialogues, %d messages changed in %d dialogues, %d dropped, removed: %s",
        cfg.voho_id,
        len(rows),
        changed_msgs,
        changed_rows,
        dropped,
        dict(terms.most_common()),
    )
    for b, a in random.Random(cfg.seed).sample(examples, min(6, len(examples))):
        log.info("- %s\n+ %s", b[:110], a[:110])
    path = jsonl.write(out_dir / f"{cfg.voho_clean_repo}.jsonl", cleaned)
    if cfg.push:
        card_data = hub.api().dataset_info(cfg.voho_id).card_data
        license_name = (card_data.to_dict() if card_data else {}).get("license", "same as the original")
        card = (
            f"---\nlanguage:\n- ar\nlicense: {license_name}\npretty_name: Voho Saudi dialogues (clean)\n---\n"
            f"# Voho Saudi dialogues (clean)\n\nCopy of [{cfg.voho_id}](https://huggingface.co/datasets/{cfg.voho_id}) "
            f"(license: {license_name}) with flirting words removed from {changed_msgs} messages: "
            f"{'، '.join(cfg.voho_flirt_terms)}. The rest of each message is unchanged.\n"
        )
        repo_id = hub.push(
            cfg.voho_clean_repo, {"data/train.jsonl": path, "README.md": card}, f"clean copy of {cfg.voho_id}"
        )
        log.info("pushed %s", repo_id)


def run(cfg: Config) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_science(cfg, out_dir)
    clean_voho(cfg, out_dir)
