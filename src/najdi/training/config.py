from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoraSettings:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05


@dataclass
class Config:
    mix_repo: str = ""
    model: str = "Qwen/Qwen3.5-9B"
    model_repo: str = ""
    mode: str = "auto"
    lora: LoraSettings = field(default_factory=LoraSettings)
    learning_rate: float = 1e-4
    warmup_frac: float = 0.03
    epochs: int = 2
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 2048
    eval_steps: int = 100
    seed: int = 1234
    groups: dict[str, str] = field(
        default_factory=lambda: {
            "helpful": "helpful",
            "identity": "identity",
            "science": "science",
            "math_logic": "math_logic",
            "voho_everyday": "voho",
            "voho_enterprise": "voho",
            "fusha": "fusha",
        }
    )
    final_eval: bool = False
    push_merged: bool = False
    noncommercial_data: bool = True
    output_dir: str = "runs/train"


@dataclass
class CompareConfig:
    mix_repo: str = ""
    candidates: list[dict] = field(default_factory=list)
    max_seq_len: int = 2048
    memory_test_batch: int = 4
    lora_r: int = 32
    delete_after: bool = True
    output_dir: str = "runs/compare"
