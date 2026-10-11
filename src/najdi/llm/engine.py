from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    model: str = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.85
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 64
    kv_cache_dtype: str = "auto"


class Engine:
    def __init__(self, cfg: EngineConfig, seed: int):
        from transformers import AutoTokenizer
        from vllm import LLM

        args = dict(
            model=cfg.model,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_seqs=cfg.max_num_seqs,
            kv_cache_dtype=cfg.kv_cache_dtype,
            seed=seed,
            reasoning_parser="qwen3",
        )
        try:
            self.llm = LLM(**args, language_model_only=True)
        except TypeError as exc:
            if "language_model_only" not in str(exc):
                raise
            self.llm = LLM(**args, limit_mm_per_prompt={"image": 0, "video": 0})
        rendered = AutoTokenizer.from_pretrained(cfg.model).apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="low",
        )
        self.has_low_effort = "Reasoning effort is set to low" in rendered
        self.model = cfg.model

    def chat(self, convs: list[list[dict]], params, thinking: str = "off") -> list[tuple[str, str]]:
        if thinking == "low" and not self.has_low_effort:
            raise ValueError(f"{self.model} has no reasoning-effort setting; set thinking to off")
        kwargs = {"enable_thinking": thinking != "off"}
        if thinking == "low":
            kwargs["reasoning_effort"] = "low"
        outs = self.llm.chat(convs, params, use_tqdm=True, chat_template_kwargs=kwargs)
        return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outs]

    @staticmethod
    def sampling(**kwargs):
        from vllm import SamplingParams

        return SamplingParams(**kwargs)

    @staticmethod
    def json_sampling(schema: dict, sampling: dict, extra_tokens: int = 0):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            **{**sampling, "max_tokens": sampling["max_tokens"] + extra_tokens},
            structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True),
        )

    @staticmethod
    def choice_sampling(choices: list[str]):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            temperature=0.0, max_tokens=16, structured_outputs=StructuredOutputsParams(choice=choices)
        )
