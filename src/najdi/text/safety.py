from __future__ import annotations

from najdi.text import lexicon
from najdi.text.arabic import normalize, term_hits, terms

_UNI, _BI = terms(lexicon.SENSITIVE_TERMS)
_IGNORE = [normalize(p) for p in lexicon.SENSITIVE_IGNORE]


def sensitive_hits(text: str) -> set[str]:
    text = normalize(text)
    for phrase in _IGNORE:
        text = text.replace(phrase, " ")
    return term_hits(text, _UNI, _BI)
