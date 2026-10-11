from __future__ import annotations

import math
import re

from najdi.chatdata import assets
from najdi.chatdata.config import Config
from najdi.chatdata.places import Places
from najdi.text.arabic import normalize, phrase_hits, phrases, stem_forms, word_count, words
from najdi.text.checks import MARKDOWN, NUMBERED, foreign_script
from najdi.text.dialect import check_per_chat
from najdi.text.openers import PRAISE, PRAISE_NEARBY, opener
from najdi.text.safety import sensitive_hits

FILTERS = [
    "format",
    "script",
    "markdown",
    "length",
    "sensitive",
    "flirting",
    "filler_opener",
    "personal_life",
    "numbers",
    "identity_claims",
    "places",
    "najdi_words",
    "dialect_clf",
    "dedup",
    "judge",
    "opener_variety",
]

INTIMATE = phrases(assets.INTIMATE_TERMS)
PERSONAL = phrases(assets.PERSONAL_TERMS)
COMPANIES = phrases(assets.COMPANY_TERMS)
UNLISTED = phrases(assets.UNLISTED_PLACES)
LANDMARKS = {normalize(w) for w in assets.LANDMARK_WORDS}
GREET_WORDS = {normalize(w) for w in assets.GREET_WORDS}

_GREETING_START = re.compile(r"^\W*(?:يا هلا|هلا والله|هلا وغلا|اهلا وسهلا|اهلين|يا مرحبا|مرحبا|هلا)(?![^\W\d_])")
_DIGITS = str.maketrans(
    "".join(map(chr, range(0x0660, 0x066A))) + "".join(map(chr, range(0x06F0, 0x06FA))), "01234567890123456789"
)
_PHONE = re.compile(r"\+?\d[\d \t-]{5,}\d")
_PRICE = re.compile(r"(\d[\d,.]*)\s*(?:-|الى|إلى|او|أو|لين)?\s*(\d[\d,.]*)?\s*(?:ريال|﷼|ر\.س)")
_SHORT_NUMBER = re.compile(r"(?<![\d.])\d{3,4}(?![\d.])")
_COMPANY_EN = re.compile(
    r"(?i)\b(?:openai|google|alibaba|qwen|meta|microsoft|anthropic|chatgpt|gpt|gemini|deepseek|mistral|llama|claude)\b"
)
_MY_NAME = re.compile(r"(?<![^\W\d_])اسمي\s+(?!مو|ما|مب|ليس)\S+")


def _generic_words() -> set[str]:
    generic = {normalize(w) for w in assets.GENERIC_WORDS} | LANDMARKS
    for city in [*assets.CITIES, *set(assets.CITIES.values()), "مكة", "المدينة", "السعودية", "المملكة"]:
        generic |= stem_forms(normalize(city))
    return generic


GENERIC = _generic_words()


def filler_opener(reply: str, user_msg: str, subset: str) -> str | None:
    if PRAISE.match(reply) or PRAISE_NEARBY.match(reply) or re.match(r"^\W*يا سلام", normalize(reply)):
        return "praise"
    greeted = {f for w in words(user_msg) for f in stem_forms(w)} & GREET_WORDS
    if subset != "identity" and _GREETING_START.match(normalize(reply)) and not greeted:
        return "greeting"
    return None


def bad_numbers(t: dict, msgs: list[dict]) -> list[str]:
    user_text = " ".join(m["content"] for m in msgs if m["role"] == "user").translate(_DIGITS)
    user_numbers = set(re.findall(r"\d+", user_text))
    bad = []
    for m in msgs:
        if m["role"] != "assistant":
            continue
        reply = m["content"].translate(_DIGITS)
        for x in _PHONE.findall(reply):
            digits = re.sub(r"\D", "", x)
            if len(digits) >= 7 and digits not in user_numbers:
                bad.append(x.strip())
        if t["topic"] != "budget_saving":
            for pair in _PRICE.findall(reply):
                bad += [x for x in (p.replace(",", "").rstrip(".") for p in pair) if x and x not in user_numbers]
        if t["subset"] != "helpful":
            bad += [x for x in _SHORT_NUMBER.findall(reply) if x not in user_numbers and not 1900 <= int(x) <= 2100]
    return bad


def dedup_text(subset: str, msgs: list[dict]) -> str:
    return " ".join(m["content"] for m in msgs if subset == "identity" or m["role"] == "user")


def row_openers(msgs: list[dict]) -> set[str]:
    return {opener(m["content"]) for m in msgs if m["role"] == "assistant"}


class RuleFilter:
    def __init__(self, cfg: Config, places: Places):
        self.cfg = cfg
        self.places = places

    def opener_cap(self, subset: str) -> int:
        return max(2, math.floor(self.cfg.filters.opener_max_frac * self.cfg.targets[subset]))

    def _place_at(self, forms: list[set[str]], i: int) -> bool:
        return any(
            i + len(p) <= len(forms) and all(p[j] in forms[i + j] for j in range(len(p))) for p in self.places.phrases
        )

    def unlisted_places(self, msgs: list[dict]) -> list[str]:
        text = "\n".join(m["content"] for m in msgs)
        tokens = words(text)
        forms = [stem_forms(w) for w in tokens]
        found = set()
        for i in range(len(tokens) - 1):
            if (
                not forms[i] & LANDMARKS
                or forms[i + 1] & GENERIC
                or self._place_at(forms, i)
                or self._place_at(forms, i + 1)
            ):
                continue
            if tokens[i + 1][:1] in "يتن" and len(tokens[i + 1]) >= 4:
                continue
            found.add(f"{tokens[i]} {tokens[i + 1]}")
        listed = {" ".join(p) for p in self.places.phrases}
        return sorted(found | {h for h in phrase_hits(text, UNLISTED) if h not in listed})

    def identity_claims(self, replies: list[str]) -> list[str]:
        bad = []
        for r in replies:
            bad += sorted(phrase_hits(r, COMPANIES)) + _COMPANY_EN.findall(r)
            if not self.cfg.assistant_name:
                bad += _MY_NAME.findall(normalize(r))
        return bad

    def check(self, t: dict, msgs: list[dict]) -> tuple[str | None, list | None]:
        f = self.cfg.filters
        subset = t["subset"]
        text = "\n".join(m["content"] for m in msgs)
        replies = [m["content"] for m in msgs if m["role"] == "assistant"]
        if foreign_script(text, f.max_english_run):
            return "script", None
        if MARKDOWN.search(text) or NUMBERED.search(text):
            return "markdown", None
        lo, hi = f.assist_words[subset]
        for m in msgs:
            n = word_count(m["content"])
            ok = lo <= n <= hi if m["role"] == "assistant" else f.user_words[0] <= n <= f.user_words[1]
            if not ok:
                return "length", None
        checks = [
            ("sensitive", lambda: sensitive_hits(text)),
            ("flirting", lambda: set().union(*(phrase_hits(r, INTIMATE) for r in replies))),
            (
                "filler_opener",
                lambda: [
                    hit
                    for i in range(1, len(msgs), 2)
                    if (hit := filler_opener(msgs[i]["content"], msgs[i - 1]["content"], subset))
                ],
            ),
            ("personal_life", lambda: set().union(*(phrase_hits(r, PERSONAL, claims=True) for r in replies))),
            ("numbers", lambda: bad_numbers(t, msgs)),
            ("identity_claims", lambda: self.identity_claims(replies)),
            ("places", lambda: self.unlisted_places(msgs) if subset == "helpful" else []),
        ]
        for name, check in checks:
            hits = check()
            if hits:
                return name, sorted(hits)
        ok = check_per_chat(msgs, f.min_saudi_per_100w, f.min_saudi_per_reply, f.najdi_min_words, f.max_fusha_ratio)
        return (None, None) if ok else ("najdi_words", None)
