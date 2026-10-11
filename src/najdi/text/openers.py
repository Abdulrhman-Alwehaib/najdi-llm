from __future__ import annotations

import re
from collections import Counter

from najdi.text.arabic import HARAKAT, WORD_END, alternation, loose, words

SEPARATORS = r"[\s،,.!؟?:؛…-]*"
LEAD = (
    alternation(
        [
            "هلا والله",
            "يا هلا والله",
            "يا هلا",
            "هلا",
            "حلا",
            "أهلا وسهلا",
            "أهلا",
            "أهلين",
            "حياك الله",
            "حياك",
            "أبشر",
            "أبشري",
            "والله",
            "وربي",
            "يا سلام",
            "ما شاء الله",
            "الحمد لله",
            "الله",
            "واو",
            "أوه",
            "أووه",
            "أه",
            "طيب",
            "زين",
            "شف",
            "شفت",
            "شفتي",
            "شوف",
        ]
    )
    + WORD_END
    + r"(?:\s+(?:فيك|بك|بكم|فيكم))?"
)
VOCATIVES = [
    "غالي",
    "الغالي",
    "غالية",
    "الغالية",
    "ولدي",
    "بنتي",
    "عمي",
    "خالي",
    "خوي",
    "أخوي",
    "أختي",
    "أمي",
    "يمه",
    "يبه",
    "أبوي",
    "أستاذ",
    "أستاذي",
    "أستاذة",
    "أستاذتي",
    "بطل",
    "البطل",
    "شيخ",
    "الشيخ",
    "طويل العمر",
    "الطيب",
    "الطيبة",
    "الذيب",
    "صاحبي",
    "دكتور",
    "دكتورة",
    "حبيبي",
    "حبيبتي",
    "قلبي",
    "عمري",
    "روحي",
    "بعد عمري",
    "عيوني",
    "القمر",
]
VOCATIVE = r"يا\s*" + alternation(VOCATIVES) + WORD_END
INTENSIFIER = alternation(["جدا", "جدًا", "مرة", "وايد", "حيل", "كثير", "بصراحة", "والله", "فعلا", "فعلًا", "صراحة"])
QUESTION = (
    r"(?:وش\s+)?(?:(?:هذا|هذاك|هذي|هذه)\s+)?(?:ها|هال|ال)?(?:"
    + loose("سؤال")
    + "|"
    + loose("سوال")
    + r")(?:ك|كم|كي|ج|ه|ها|هم|كن)?"
    + HARAKAT
    + WORD_END
)
ADJECTIVE = (
    alternation(
        [
            "في محله",
            "بمحله",
            "في الصميم",
            "حلو",
            "زين",
            "ذكي",
            "دقيق",
            "مهم",
            "منطقي",
            "جميل",
            "رائع",
            "ممتاز",
            "جيد",
            "عميق",
            "رهيب",
            "يجنن",
            "ممتع",
            "كفو",
            "حلو مرة",
            "واجد حلو",
        ]
    )
    + WORD_END
)
CORE = (
    rf"(?:(?:و?ترى|صراحة|بصراحة|والله)\s+)?"
    rf"(?:{QUESTION}(?:\s*{VOCATIVE})?(?:\s+{INTENSIFIER})*\s+{ADJECTIVE}(?:\s+{INTENSIFIER})*"
    rf"|{ADJECTIVE}(?:\s+{INTENSIFIER})*\s+{QUESTION}"
    rf"|سألت(?:ي|وا)?\s+(?:سؤال|سوال)\s+(?:{INTENSIFIER}\s+)*{ADJECTIVE})"
)
PRAISE = re.compile(
    rf"^\s*(?:{LEAD}{SEPARATORS})*(?:{VOCATIVE}{SEPARATORS})?{CORE}(?:{SEPARATORS}{VOCATIVE})?{SEPARATORS}"
)
PRAISE_NEARBY = re.compile(
    rf"^(?:\S+\s+){{0,4}}?(?:\S*(?:سؤال|سوال)\S*\s+(?:\S+\s+)?{ADJECTIVE}|{ADJECTIVE}\s+\S*(?:سؤال|سوال))"
)
AND_FIX = re.compile(
    r"^و(?=(?:ترى|هال|هاي|هذي|هذا|هذه|واضح|فعلا|كثير|الحين|خل|بصراحة|يعني|الفكرة|السر|الموضوع|النقطة"
    r"|هو|هي|الجواب|السبب|اللي|لأن|لان|عشان|خذ|شوف|تخيل)\S*)"
)


def filler_pattern(phrases: list[str]) -> re.Pattern | None:
    if not phrases:
        return None
    return re.compile(
        rf"^\s*(?:{LEAD}{SEPARATORS})*{alternation(phrases)}{WORD_END}(?:{SEPARATORS}{VOCATIVE})?{SEPARATORS}"
    )


def cut_opener(text: str, extra: re.Pattern | None = None) -> tuple[str, bool]:
    for pattern in (PRAISE, extra):
        match = pattern.match(text) if pattern else None
        if not match:
            continue
        rest = text[match.end() :]
        runs_on = not re.search(r"[،,.!؟?:؛…\n-]\s*$", match.group(0)) and not re.match(r"(?:و|بس\s|لأن|لان)", rest)
        if pattern is PRAISE and runs_on:
            sentence = re.match(r"[^.!؟?\n]*(?:[.!؟?]+|\n|$)", rest)
            rest = rest[sentence.end() :] if len(sentence.group(0).split()) <= 6 else ""
        rest = re.sub(r"^بس\s+", "", AND_FIX.sub("", rest.lstrip(" \t\n،,.!؟?:؛…-")))
        if len(rest.split()) >= 3:
            return rest, True
    return text, False


def opener(text: str, n: int = 4) -> str:
    return " ".join(words(text)[:n])


def opener_counts(replies: list[str], k: int = 20) -> list[tuple[str, int]]:
    return Counter(opener(t) for t in replies).most_common(k)
