from __future__ import annotations

from dataclasses import dataclass, field

from najdi.llm.engine import EngineConfig


@dataclass
class FilterConfig:
    user_words: list[int] = field(default_factory=lambda: [1, 90])
    assist_words: dict[str, list[int]] = field(default_factory=lambda: {"helpful": [5, 95], "identity": [2, 70]})
    min_saudi_per_100w: float = 5.0
    min_saudi_per_reply: int = 1
    najdi_min_words: int = 6
    max_fusha_ratio: float = 0.2
    max_english_run: int = 3
    dialect_classifier: str = "IbrahimAmin/marbertv2-arabic-written-dialect-classifier"
    classifier_min_words: int = 5
    dedup_jaccard: float = 0.6
    opener_max_frac: float = 0.02


@dataclass
class Config:
    engine: EngineConfig = field(default_factory=EngineConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    thinking: str = "low"
    judge_thinking: str = "low"
    thinking_extra_tokens: int = 2000
    sampling: dict = field(
        default_factory=lambda: {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 20,
            "presence_penalty": 0.0,
            "max_tokens": 2500,
        }
    )
    judge_sampling: dict = field(
        default_factory=lambda: {"temperature": 0.3, "top_p": 0.95, "top_k": 20, "max_tokens": 600}
    )
    pilot: bool = True
    pilot_rows: int = 50
    targets: dict[str, int] = field(default_factory=lambda: {"helpful": 1500, "identity": 200})
    parts: dict[str, list[str]] = field(default_factory=lambda: {"A": ["helpful"], "B": ["identity"]})
    batch_tasks: int = 120
    push_every: int = 300
    telegram_every: int = 100
    seed: int = 1234
    max_attempts_factor: int = 15
    assistant_name: str = ""
    creator_name: str = ""
    repo_id: str = ""
    voho_id: str = "VohoAI/voho-saudi-dialogues"
    places_per_prompt: int = 8
    place_summary_words: int = 40
    anchor_max_messages: int = 8
    output_dir: str = "runs/chat-data"
