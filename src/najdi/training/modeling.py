from __future__ import annotations

import re

from najdi.env import hf_token

_NOT_LANGUAGE = ("vision", "visual", "audio", "multi_modal_projector", "embed_vision", "embed_audio", "lm_head")


def gpu_gb() -> float:
    import torch

    return torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0


def parameter_count(model_id: str) -> int:
    from huggingface_hub import get_safetensors_metadata

    return sum(get_safetensors_metadata(model_id, token=hf_token()).parameter_count.values())


def bf16_fits(model_id: str, margin_gb: float = 8.0) -> bool:
    return parameter_count(model_id) * 2 / 1e9 + margin_gb <= gpu_gb()


def load_model(model_id: str, quantize: bool = False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    kwargs = {"token": hf_token(), "dtype": torch.bfloat16, "device_map": {"": 0}}
    if quantize:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if AutoConfig.from_pretrained(model_id, token=hf_token()).model_type == "gemma2":
        kwargs["attn_implementation"] = "eager"
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


def lora_targets(model) -> list[str]:
    import torch

    return [
        n
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and re.search(r"(^|\.)layers\.\d+\.", n)
        and not any(s in n for s in _NOT_LANGUAGE)
    ]
