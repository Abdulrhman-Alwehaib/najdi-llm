from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from najdi import jsonl
from najdi.science import assets
from najdi.science.config import DomainConfig, WikiConfig
from najdi.text.arabic import normalize, strip_marks, term_hits, word_count, words
from najdi.text.safety import sensitive_hits

log = logging.getLogger(__name__)

_BAD_TITLE = re.compile(r"[0-9٠-٩A-Za-z]")
_FIRST_WORDS = {normalize(w) for w in assets.NON_SCIENCE_FIRST_WORDS}
_PAREN_WORDS = {normalize(w) for w in assets.NON_SCIENCE_PAREN_WORDS}
_CATEGORY_WORDS = {normalize(w) for w in assets.CATEGORY_WORDS}
_STOP_HEADERS = {normalize(h) for h in assets.STOP_HEADERS}
_BIO = re.compile(r"(?:^|\s)(?:ولد|ولدت|توفي|توفيت|مواليد)(?:\s|$)|^[^.]{0,120}\([^)]{0,60}\d{3,4}")
_IS_A = re.compile(r"(?:^|\s)(?:هو|هي|كان|كانت)\s+(?:" + "|".join(sorted(_CATEGORY_WORDS)) + r")(?:\s|$|،|\.)")


class KeywordIndex:
    def __init__(self, keywords: dict[str, str]):
        self.topics: dict[str, set[str]] = defaultdict(set)
        for topic, items in keywords.items():
            for item in items.split():
                self.topics[normalize(item.replace("_", " "))].add(topic)
        self.uni = {k for k in self.topics if " " not in k}
        self.bi = {tuple(k.split()) for k in self.topics if " " in k}

    def hits(self, text: str) -> set[str]:
        return term_hits(text, self.uni, self.bi)

    def topic(self, hits: set[str]) -> str:
        return Counter(t for h in hits for t in self.topics[h]).most_common(1)[0][0]


def _non_topic_lead(passage: str) -> bool:
    head = normalize(passage[:300])
    return bool(_BIO.search(head) or _IS_A.search(head) or set(words(head)[:8]) & _CATEGORY_WORDS)


def _cut_sentences(line: str, k: int) -> str:
    s = " ".join(line.split()[:k])
    end = max(s.rfind("."), s.rfind("؟"), s.rfind("!"))
    return s[: end + 1] if end > 0 else ""


def extract_passage(text: str, lo: int, hi: int) -> tuple[str, int]:
    paras, total = [], 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        n = word_count(line)
        if n < 8 and not line.endswith((".", "؟", "!", ":")):
            if total >= lo or normalize(line) in _STOP_HEADERS:
                break
            continue
        if total + n > hi:
            if total >= lo:
                break
            line = _cut_sentences(line, hi - total)
            n = word_count(line)
            if not n:
                break
        paras.append(line)
        total += n
        if total >= hi:
            break
    return strip_marks("\n\n".join(paras)), total


def candidate(article: dict, index: KeywordIndex, cfg: WikiConfig) -> dict | None:
    title = article["title"]
    if _BAD_TITLE.search(title):
        return None
    nt = normalize(title)
    if not nt.split() or nt.split()[0] in _FIRST_WORDS:
        return None
    if "(" in nt and any(w in _PAREN_WORDS for w in words(nt[nt.index("(") :])):
        return None
    lo, hi = cfg.passage_words
    passage, n = extract_passage(article["text"][:8000], lo, hi)
    if n < lo or _non_topic_lead(passage):
        return None
    hits = index.hits(title + " " + passage)
    if len(hits) < cfg.min_keyword_hits:
        return None
    return {
        "id": article["id"],
        "url": article["url"],
        "title": title,
        "passage": passage,
        "kw_hits": len(hits),
        "kw_topic": index.topic(hits),
    }


def keyword_pass(cfg: WikiConfig, index: KeywordIndex, seed: int, path: Path) -> list[dict]:
    if path.exists():
        return jsonl.read(path)
    from datasets import load_dataset
    from tqdm.auto import tqdm

    wiki = load_dataset(cfg.dataset, cfg.subset, split="train", streaming=True)
    groups: dict[str, list[dict]] = defaultdict(list)
    for article in tqdm(wiki, total=cfg.articles, desc="keyword pass"):
        found = candidate(article, index, cfg)
        if found:
            groups[found["kw_topic"]].append(found)
            if cfg.max_candidates and sum(map(len, groups.values())) >= cfg.max_candidates:
                break
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    cands = [c for row in zip_longest(*(groups[t] for t in sorted(groups))) for c in row if c]
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    jsonl.append(tmp, cands)
    tmp.replace(path)
    return cands


class PassagePool:
    def __init__(self, cands: list[dict], domain: DomainConfig, engine, labels_path: Path, chunk: int):
        self.cands = cands
        self.domain = domain
        self.engine = engine
        self.labels_path = labels_path
        self.chunk = chunk
        self.choices = [*domain.topics, "none"]
        self.version = hashlib.md5(domain.label_prompt.encode("utf-8")).hexdigest()[:8]
        self.labels = {r["id"]: r["label"] for r in jsonl.read(labels_path) if r.get("version") == self.version}

    def _ask(self, cands: list[dict]) -> list[str]:
        convs = [
            [
                {
                    "role": "user",
                    "content": self.domain.label_prompt.format(title=c["title"], passage=c["passage"][:1500]),
                }
            ]
            for c in cands
        ]
        outs = self.engine.chat(convs, self.engine.choice_sampling(self.choices))
        return [text.strip() if text.strip() in self.choices else "none" for text, _ in outs]

    def label_more(self) -> bool:
        todo = [c for c in self.cands if c["id"] not in self.labels][: self.chunk]
        if not todo:
            return False
        sensitive = [bool(sensitive_hits(c["title"] + " " + c["passage"])) for c in todo]
        ask = [c for c, s in zip(todo, sensitive) if not s]
        log.info("labelling %d passages (%d on sexual topics skipped)", len(ask), len(todo) - len(ask))
        answers = iter(self._ask(ask) if ask else [])
        labels = ["sensitive" if s else next(answers) for s in sensitive]
        jsonl.append(
            self.labels_path,
            [
                {"id": c["id"], "label": label, "labeller": self.engine.model, "version": self.version}
                for c, label in zip(todo, labels)
            ],
        )
        self.labels.update((c["id"], label) for c, label in zip(todo, labels))
        return True

    def _free(self, used: set[str]) -> dict[str, list[dict]]:
        by_topic = defaultdict(list)
        for c in self.cands:
            label = self.labels.get(c["id"])
            if label in self.domain.topics and c["id"] not in used:
                by_topic[label].append(c)
        return by_topic

    def take(self, n: int, used: set[str], kept_per_topic: Counter) -> list[dict]:
        free = self._free(used)
        while sum(map(len, free.values())) < 2 * n and self.label_more():
            free = self._free(used)
        counts = Counter(kept_per_topic)
        picked = []
        while len(picked) < n and any(free.values()):
            topic = min((t for t in self.domain.topics if free[t]), key=lambda t: counts[t])
            picked.append({**free[topic].pop(0), "label": topic})
            counts[topic] += 1
        return picked
