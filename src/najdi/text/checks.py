from __future__ import annotations

import re


def _ranges(pairs: list[tuple[int, int]]) -> re.Pattern:
    return re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in pairs) + "]")


CJK = _ranges(
    [
        (0x3000, 0x303F),
        (0x3040, 0x30FF),
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xAC00, 0xD7AF),
        (0xF900, 0xFAFF),
        (0xFF00, 0xFFEF),
    ]
)
OTHER_SCRIPTS = _ranges([(0x0400, 0x04FF), (0x0590, 0x05FF), (0x0900, 0x097F), (0x0E00, 0x0E7F)])
ENGLISH_WORD = re.compile(r"[A-Za-z](?:[A-Za-z'’-]*[A-Za-z])?")
MARKDOWN = re.compile(
    r"\*\*|__|```|^\s{0,3}#{1,6}\s|^\s*[-*•●▪]\s+|^\s*\|.*\|\s*$|\$[^$\n]+\$|\\\(|\\\[|\\frac|\\times|\\cdot", re.M
)
NUMBERED = re.compile(r"^\s*\d+\s*[.)-]\s", re.M)


def longest_english_run(text: str) -> int:
    best = run = 0
    for token in text.split():
        token = token.strip(".,;:!?()[]{}\"'«»،؛؟")
        run = run + 1 if ENGLISH_WORD.fullmatch(token) else 0
        best = max(best, run)
    return best


def foreign_script(text: str, max_english_run: int) -> bool:
    return bool(CJK.search(text) or OTHER_SCRIPTS.search(text) or longest_english_run(text) > max_english_run)
