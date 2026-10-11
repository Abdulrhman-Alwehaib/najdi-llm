from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

SOURCES = ["helpful", "identity", "science", "math_logic", "voho_everyday", "voho_enterprise", "fusha"]
CLEAN_STEPS = [
    "broken_roles",
    "fusha_short",
    "fusha_translation",
    "fusha_external",
    "sensitive",
    "non_najdi",
    "voho_intimate",
    "voho_personal",
    "exact_duplicate",
    "near_duplicate",
    "too_long",
]


@dataclass
class Config:
    chat_repo: str = ""
    helpful_repeat: int = 2
    science_repos: list[str] = field(default_factory=list)
    science_n: int = 1500
    always_keep_topics: list[str] = field(default_factory=lambda: ["medical"])
    voho_id: str = ""
    voho_everyday_n: int = 300
    voho_enterprise_n: int = 700
    fusha_source: str = "cidar"
    fusha_n: int = 400
    fusha_system_prompt: str = "رد بالفصحى"
    fusha_min_output_words: int = 20
    fusha_min_instruction_words: int = 3
    fusha_max_latin_words: int = 2
    non_najdi_words: list[str] = field(default_factory=list)
    abshir_cut_frac: float = 0.5
    voho_intimate: list[str] = field(default_factory=list)
    voho_family_role: list[str] = field(default_factory=list)
    voho_personal: list[str] = field(default_factory=list)
    screen_sensitive: bool = True
    near_dup_jaccard: float = 0.7
    tokenizer: str = "Qwen/Qwen3.5-9B-Base"
    max_seq_len: int = 2048
    test_n: dict[str, int] = field(default_factory=lambda: {"science_logic": 50, "voho": 50, "fusha": 20})
    val_frac: float = 0.05
    seed: int = 1234
    repo_id: str = ""
    push: bool = True
    output_dir: str = "runs/mix"


@dataclass
class Report:
    loaded: Counter = field(default_factory=Counter)
    removed: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    fixed: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    abshir: dict[str, tuple[int, int]] = field(default_factory=dict)
    voho_system_rows: int = 0

    def drop_where(self, rows: list[dict], step: str, is_bad) -> list[dict]:
        kept = []
        for r in rows:
            if is_bad(r):
                self.removed[step][r["source"]] += 1
            else:
                kept.append(r)
        return kept


def counts_line(rows: list[dict]) -> str:
    c = Counter(r["source"] for r in rows)
    return f"{len(rows)} rows (" + ", ".join(f"{s} {c[s]}" for s in SOURCES) + ")"
