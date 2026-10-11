from __future__ import annotations

from collections.abc import Iterable

from najdi.text import lexicon
from najdi.text.arabic import normalize, words


def _with_prefixes(items: Iterable[str], drop: Iterable[str] = ()) -> set[str]:
    out = set()
    for w in items:
        w = normalize(w)
        out |= {w, "و" + w, "ف" + w}
    return out - set(drop)


SAUDI = _with_prefixes(lexicon.SAUDI_MARKERS)
FUSHA = _with_prefixes(lexicon.FUSHA_MARKERS, drop=lexicon.NOT_FUSHA)
_HAL_EXCEPT = set(lexicon.HAL_EXCEPT)


def marker_counts(text: str) -> tuple[int, int]:
    saudi = fusha = 0
    for w in words(text):
        if w in SAUDI or (w.startswith("هال") and len(w) > 4 and w not in _HAL_EXCEPT):
            saudi += 1
        elif w in FUSHA:
            fusha += 1
    return saudi, fusha


def check_per_message(
    msgs: list[dict], min_per_100w: float, min_per_reply: int, max_fusha_ratio: float
) -> tuple[bool, int, int]:
    ok, saudi_total, fusha_total = True, 0, 0
    for m in msgs:
        saudi, fusha = marker_counts(m["content"])
        saudi_total, fusha_total = saudi_total + saudi, fusha_total + fusha
        if fusha > max_fusha_ratio * saudi:
            ok = False
        if m["role"] == "assistant" and (
            saudi < min_per_reply or 100 * saudi / max(len(words(m["content"])), 1) < min_per_100w
        ):
            ok = False
    return ok, saudi_total, fusha_total


def check_per_chat(
    msgs: list[dict], min_per_100w: float, min_per_reply: int, min_words: int, max_fusha_ratio: float
) -> bool:
    saudi = fusha = reply_saudi = reply_words = 0
    for m in msgs:
        s, f = marker_counts(m["content"])
        saudi, fusha = saudi + s, fusha + f
        if m["role"] == "assistant":
            n = len(words(m["content"]))
            reply_saudi, reply_words = reply_saudi + s, reply_words + n
            if n >= min_words and s < min_per_reply:
                return False
    return fusha <= max_fusha_ratio * saudi and 100 * reply_saudi / max(reply_words, 1) >= min_per_100w
