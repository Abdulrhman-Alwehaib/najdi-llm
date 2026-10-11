from __future__ import annotations

import re

from najdi.science import assets
from najdi.science.config import FilterConfig
from najdi.text.arabic import normalize, word_count, words
from najdi.text.checks import MARKDOWN, foreign_script
from najdi.text.safety import sensitive_hits

_DIGITS = str.maketrans(
    "".join(map(chr, range(0x0660, 0x066A))) + "".join(map(chr, range(0x06F0, 0x06FA))) + chr(0x066B) + chr(0x2212),
    "01234567890123456789.-",
)
_NUMBER = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")
_SOURCE_MENTION = re.compile("|".join(re.escape(normalize(w)) for w in assets.SOURCE_MENTIONS))
_PERCENT_WORDS = ("بالميه", "بالمئه", "في الميه", "في المئه")


def cheap_filter(msgs: list[dict], filters: FilterConfig) -> str | None:
    text = "\n".join(m["content"] for m in msgs)
    if foreign_script(text, filters.max_english_run):
        return "script"
    if MARKDOWN.search(text):
        return "markdown"
    for m in msgs:
        lo, hi = filters.assist_words if m["role"] == "assistant" else filters.user_words
        if not lo <= word_count(m["content"]) <= hi:
            return "length"
    if _SOURCE_MENTION.search(normalize(text)):
        return "source_mention"
    if sensitive_hits(text):
        return "sensitive"
    return None


def numbers_in(text: str) -> list[float]:
    text = text.translate(_DIGITS).replace(chr(0x066C), "")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    out = []
    for m in _NUMBER.finditer(text):
        a, _, b = m.group().replace(" ", "").partition("/")
        if not b:
            out.append(float(a))
        elif float(b):
            out.append(float(a) / float(b))
    return out


def _close(x: float, y: float) -> bool:
    if x.is_integer() and y.is_integer():
        return x == y
    return abs(x - y) <= max(0.011, 2e-3 * max(abs(x), abs(y)))


def _has_percent(text: str) -> bool:
    return "%" in text or "٪" in text or any(w in normalize(text) for w in _PERCENT_WORDS)


def answers_match(a: str, b: str) -> bool:
    na, nb = numbers_in(a), numbers_in(b)
    if na and nb:
        if len(na) == len(nb) and all(_close(x, y) for x, y in zip(sorted(na), sorted(nb))):
            return True
        if len(na) == 1 and len(nb) == 1 and _has_percent(a) != _has_percent(b):
            return _close(na[0], nb[0] * 100) or _close(na[0] * 100, nb[0])
        return False
    if not na and not nb:
        ka, kb = " ".join(words(a)), " ".join(words(b))
        return bool(ka and kb) and (ka == kb or (min(len(ka), len(kb)) >= 3 and (ka in kb or kb in ka)))
    return False


def answer_in_reply(answer: str, reply: str) -> bool:
    have = numbers_in(reply)
    return all(any(_close(x, y) for y in have) for x in numbers_in(answer))


def final_answer(text: str, finish_reason: str) -> str | None:
    if finish_reason == "length":
        return None
    hits = re.findall(r"FINAL\s*[:：]\s*(.+)", text.split("</think>")[-1])
    return hits[-1].strip(" *`$") if hits else None
