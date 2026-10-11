from __future__ import annotations

import json
from collections import Counter

from najdi.training.config import Config


def model_card(
    cfg: Config,
    repo_id: str,
    mix_repo: str,
    eot: str,
    quantized: bool,
    merged: bool,
    counts: dict[str, Counter],
    losses: dict[str, float] | None,
    minutes: float,
) -> str:
    groups = sorted(set(cfg.groups.values()), key=list(cfg.groups.values()).index)
    loss = (lambda g: f"{losses[g]:.4f}") if losses else (lambda g: "-")
    data_rows = "\n".join(f"| {g} | {counts['train'][g]:,} | {counts['validation'][g]:,} | {loss(g)} |" for g in groups)
    method = "QLoRA (4-bit NF4)" if quantized else "LoRA, bf16"
    nc = (
        "\n**Non-commercial research use only.** The training mix includes CIDAR (CC BY-NC 4.0).\n"
        if cfg.noncommercial_data
        else ""
    )
    if merged:
        usage = f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{repo_id}", token=True)
model = AutoModelForCausalLM.from_pretrained("{repo_id}", token=True, dtype="bfloat16", device_map="auto")
```"""
    else:
        usage = f"""```python
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
path = snapshot_download("{repo_id}", allow_patterns=["adapter/*"], token=True)
tok = AutoTokenizer.from_pretrained("{cfg.model}")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{cfg.model}", dtype="bfloat16", device_map="auto"), path + "/adapter")
```"""
    return f"""---
language:
- ar
license: other
license_name: {"apache-2.0-with-cc-by-nc-data" if cfg.noncommercial_data else "apache-2.0"}
base_model: {cfg.model}
library_name: {"transformers" if merged else "peft"}
tags:
- saudi
- najdi
- arabic-dialect
- lora
---

# Najdi Saudi chat model

A {method} fine-tune of [`{cfg.model}`](https://huggingface.co/{cfg.model}) that chats in Najdi Saudi Arabic by default
and answers in Modern Standard Arabic when the system message is `رد بالفصحى`.
{nc}
Use the chat template in non-thinking mode (`enable_thinking=False`) and stop at {json.dumps(eot)}.

{usage}

## Repo layout

{"- root: the merged bf16 model" if merged else "- root: this card"}
- `adapter/`: the best LoRA adapter
- `best-checkpoint/`: the best adapter as training went
- `epoch-N/`: the adapter after each epoch
- `last-checkpoint/`: the newest training checkpoint, for resuming

## Training data

[`{mix_repo}`](https://huggingface.co/datasets/{mix_repo}), train and validation splits only.

| group | train rows | validation rows | validation loss |
|---|---|---|---|
{data_rows}

## Hyperparameters

| | |
|---|---|
| method | {method}, all linear layers of the language model |
| LoRA r / alpha / dropout | {cfg.lora.r} / {cfg.lora.alpha} / {cfg.lora.dropout} |
| learning rate | {cfg.learning_rate}, cosine, {cfg.warmup_frac:.0%} warmup |
| epochs | {cfg.epochs} |
| effective batch | {cfg.batch_size * cfg.grad_accum} |
| max length | {cfg.max_seq_len} tokens, no packing |
| training time | {minutes} min |
"""
