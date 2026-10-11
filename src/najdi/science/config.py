from __future__ import annotations

from dataclasses import dataclass, field

from najdi.llm.engine import EngineConfig


@dataclass
class DomainConfig:
    name: str = "science"
    title: str = "Najdi Saudi science and math conversations"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    topics: dict[str, str] = field(default_factory=dict)
    keywords: dict[str, str] = field(default_factory=dict)
    label_prompt: str = ""
    extra_rules: str = ""
    examples: list[dict] = field(default_factory=list)


@dataclass
class WikiConfig:
    dataset: str = "wikimedia/wikipedia"
    subset: str = "20231101.ar"
    articles: int = 1219201
    passage_words: list[int] = field(default_factory=lambda: [120, 380])
    min_keyword_hits: int = 3
    label_chunk: int = 1500
    max_candidates: int | None = None


@dataclass
class FilterConfig:
    assist_words: list[int] = field(default_factory=lambda: [30, 230])
    user_words: list[int] = field(default_factory=lambda: [2, 80])
    min_saudi_per_100w: float = 8.0
    min_saudi_per_reply: int = 1
    max_fusha_ratio: float = 0.2
    max_english_run: int = 4
    dialect_classifier: str = "IbrahimAmin/marbertv2-arabic-written-dialect-classifier"
    dedup_jaccard: float = 0.6
    check_answer_in_reply: bool = True


@dataclass
class Config:
    engine: EngineConfig = field(default_factory=EngineConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    wiki: WikiConfig = field(default_factory=WikiConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    thinking: str = "low"
    thinking_extra_tokens: int = 2000
    sampling: dict = field(
        default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 0.0,
            "max_tokens": 3000,
        }
    )
    verify_sampling: dict = field(
        default_factory=lambda: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 6000}
    )
    pilot: bool = True
    pilot_rows: int = 100
    target_rows: int = 4000
    science_frac: float = 0.6
    batch_tasks: int = 160
    push_every: int = 500
    telegram_every: int = 100
    seed: int = 1234
    id_prefix: str = "sci"
    repo_id: str = ""
    voho_id: str = "VohoAI/voho-saudi-dialogues"
    anchors_per_prompt: list[int] = field(default_factory=lambda: [2, 3])
    anchor_max_messages: int = 8
    user_turn_weights: dict[int, float] = field(default_factory=lambda: {2: 0.4, 3: 0.4, 4: 0.2})
    output_dir: str = "runs/science"
