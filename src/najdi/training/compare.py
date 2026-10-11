from __future__ import annotations

import gc
import json
import logging
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from najdi import hub
from najdi.env import hf_token
from najdi.notify import telegram
from najdi.text.arabic import word_count
from najdi.training.config import CompareConfig
from najdi.training.modeling import load_model, lora_targets, parameter_count
from najdi.training.templates import build_examples, choose_pad, end_of_turn, load_tokenizer

log = logging.getLogger(__name__)

GROUPS = ("saudi", "fusha")


def tokens_per_word(tok, rows: list[dict]) -> dict[str, float]:
    out = {}
    for group in GROUPS:
        texts = [m["content"] for r in rows if r["group"] == group for m in r["messages"] if m["role"] != "system"]
        tokens = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
        out[group] = tokens / sum(word_count(t) for t in texts)
    return out


def reply_perplexity(model, tok, eot: str, rows: list[dict], max_len: int) -> dict[str, tuple[float, float]]:
    import torch

    nll, tokens, words = defaultdict(float), Counter(), Counter()
    with torch.no_grad():
        for r in rows:
            for ex in build_examples(tok, r["messages"], eot, score_eot=False):
                ids = torch.tensor([ex["input_ids"][:max_len]], device="cuda")
                labels = torch.tensor([ex["labels"][:max_len]], device="cuda")
                n = int((labels[:, 1:] != -100).sum())
                if n:
                    nll[r["group"]] += model(input_ids=ids, labels=labels).loss.float().item() * n
                    tokens[r["group"]] += n
            words[r["group"]] += sum(word_count(m["content"]) for m in r["messages"] if m["role"] == "assistant")
    return {
        g: (float(torch.exp(torch.tensor(nll[g] / tokens[g]))), float(torch.exp(torch.tensor(nll[g] / words[g]))))
        for g in GROUPS
    }


def lora_step_peak_gb(model, tok, eot: str, rows: list[dict], cfg: CompareConfig) -> float:
    import torch
    from peft import LoraConfig, get_peft_model

    examples = [ex for r in rows for ex in build_examples(tok, r["messages"], eot)]
    longest = sorted(examples, key=lambda e: -len(e["input_ids"]))[: cfg.memory_test_batch]
    n = min(cfg.max_seq_len, max(len(e["input_ids"]) for e in longest))
    ids = torch.full((len(longest), n), tok.pad_token_id, device="cuda")
    labels = torch.full((len(longest), n), -100, device="cuda")
    mask = torch.zeros((len(longest), n), dtype=torch.long, device="cuda")
    for i, e in enumerate(longest):
        k = min(n, len(e["input_ids"]))
        ids[i, :k], labels[i, :k], mask[i, :k] = torch.tensor(e["input_ids"][:k]), torch.tensor(e["labels"][:k]), 1
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=2 * cfg.lora_r, target_modules=lora_targets(model), task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, lora)
    torch.cuda.reset_peak_memory_stats()
    peft_model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del peft_model
    return peak


def delete_from_cache(model_id: str) -> None:
    from huggingface_hub import constants

    shutil.rmtree(Path(constants.HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--")), ignore_errors=True)


def score(candidate: dict, rows: list[dict], cfg: CompareConfig) -> dict:
    import torch

    row = dict(candidate)
    model = None
    try:
        tok = load_tokenizer(candidate["model"])
        eot = end_of_turn(tok)
        tok.pad_token = choose_pad(tok, eot)
        row.update({f"tokens_per_word_{g}": v for g, v in tokens_per_word(tok, rows).items()})
        row["params_b"] = parameter_count(candidate["model"]) / 1e9
        model = load_model(candidate["model"])
        model.eval()
        for g, (token_ppl, word_ppl) in reply_perplexity(model, tok, eot, rows, cfg.max_seq_len).items():
            row[f"ppl_{g}"], row[f"word_ppl_{g}"] = token_ppl, word_ppl
        model.train()
        row["lora_step_peak_gb"] = lora_step_peak_gb(model, tok, eot, rows, cfg)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        log.warning("%s failed: %s", candidate["model"], row["error"])
    finally:
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        if cfg.delete_after:
            delete_from_cache(candidate["model"])
    return row


def run(cfg: CompareConfig) -> None:
    from datasets import load_dataset

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    val = load_dataset(hub.resolve(cfg.mix_repo), split="validation", token=hf_token())
    rows = [{"group": "fusha" if r["source"] == "fusha" else "saudi", "messages": r["messages"]} for r in val]
    done = set()
    if results.exists():
        done = {
            json.loads(line)["model"]
            for line in results.read_text(encoding="utf-8").splitlines()
            if "error" not in line
        }
    telegram(f"model comparison started: {len(cfg.candidates)} models, {len(done)} done.")
    for candidate in cfg.candidates:
        if candidate["model"] in done:
            continue
        t0 = time.time()
        row = score(candidate, rows, cfg)
        row["minutes"] = round((time.time() - t0) / 60, 1)
        with results.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info(json.dumps(row, ensure_ascii=False))
        telegram(
            f"{candidate['model']} scored in {row['minutes']} min" + (f": {row['error']}" if "error" in row else "")
        )
