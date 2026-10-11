from __future__ import annotations

import re
from collections.abc import Iterable

_MARK_CODES = [*range(0x0610, 0x061B), *range(0x064B, 0x0660), 0x0670, *range(0x06D6, 0x06EE), 0x0640]
_MARKS = re.compile("[" + "".join(map(chr, _MARK_CODES)) + "]")
LETTERS = f"{chr(0x0621)}-{chr(0x064A)}"
_WORD = re.compile(f"[{LETTERS}]+")
_PREFIXES_1, _PREFIXES_2 = ("و", "ف"), ("بال", "كال", "لل", "ال", "ب", "ل", "ك")

HARAKAT = "[" + chr(0x064B) + "-" + chr(0x0652) + chr(0x0640) + "]*"
WORD_END = r"(?![^\W\d_])"
WORD_START = r"(?<![^\W\d_])"


def strip_marks(text: str) -> str:
    return _MARKS.sub("", text)


def normalize(text: str) -> str:
    text = _MARKS.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    return text.replace("ى", "ي").replace("ة", "ه")


def words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def word_count(text: str) -> int:
    return len(text.split())


def stem_forms(word: str) -> set[str]:
    forms = {word}
    for rest in [word] + [word[1:] for p in _PREFIXES_1 if word.startswith(p) and len(word) > 3]:
        forms.add(rest)
        for p in _PREFIXES_2:
            if rest.startswith(p) and len(rest) - len(p) >= 3:
                forms.add(rest[len(p) :])
    return forms


def terms(items: Iterable[str]) -> tuple[set[str], set[tuple[str, ...]]]:
    normalized = {normalize(t) for t in items}
    return {t for t in normalized if " " not in t}, {tuple(t.split()) for t in normalized if " " in t}


def term_hits(text: str, uni: set[str], bi: set[tuple[str, ...]]) -> set[str]:
    forms = [stem_forms(w) for w in words(text)]
    hits = set()
    for i, fs in enumerate(forms):
        hits |= fs & uni
        if i + 1 < len(forms):
            hits |= {f"{a} {b}" for a in fs for b in forms[i + 1] if (a, b) in bi}
    return hits


def phrases(items: Iterable[str]) -> list[tuple[str, ...]]:
    return sorted({tuple(normalize(p).split()) for p in items if p.strip()})


def definite(word: str) -> bool:
    w = word[1:] if word[:1] in "وف" and len(word) > 3 else word
    return w.startswith(("ال", "بال", "لل", "كال"))


def phrase_hits(text: str, items: list[tuple[str, ...]], claims: bool = False) -> set[str]:
    tokens = words(text)
    forms = [stem_forms(w) for w in tokens]
    hits = set()
    for phrase in items:
        for i in range(len(forms) - len(phrase) + 1):
            if not all(phrase[j] in forms[i + j] for j in range(len(phrase))):
                continue
            if claims and len(phrase) == 1 and ((i and tokens[i - 1] == "يا") or definite(tokens[i])):
                continue
            hits.add(" ".join(phrase))
    return hits


def loose(phrase: str) -> str:
    out = []
    for ch in phrase:
        if ch == " ":
            out.append(r"\s*")
            continue
        if ch in "اأإآ":
            out.append("[اأإآ]" + HARAKAT)
        elif ch in "ةه":
            out.append("[ةه]" + HARAKAT)
        elif ch in "ىي":
            out.append("[ىي]" + HARAKAT)
        else:
            out.append(re.escape(ch) + HARAKAT)
    return "".join(out)


def alternation(items: Iterable[str]) -> str:
    return "(?:" + "|".join(loose(p) for p in sorted(set(items), key=len, reverse=True)) + ")"
