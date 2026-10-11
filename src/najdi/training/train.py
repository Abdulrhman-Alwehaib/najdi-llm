from __future__ import annotations

import gc
import logging
import math
import shutil
import time
from collections import Counter
from pathlib import Path

from najdi import hub
from najdi.env import hf_token
from najdi.notify import telegram
from najdi.training.callbacks import HubCheckpoints, Progress, resume_checkpoint
from najdi.training.card import model_card
from najdi.training.config import Config
from najdi.training.modeling import bf16_fits, load_model, lora_targets
from najdi.training.templates import build_examples, choose_pad, end_of_turn, load_tokenizer

log = logging.getLogger(__name__)


def to_examples(rows, tok, eot: str, max_len: int, groups: dict[str, str]) -> tuple[list[dict], int]:
    out, too_long = [], 0
    for r in rows:
        for ex in build_examples(tok, r["messages"], eot):
            if len(ex["input_ids"]) > max_len:
                too_long += 1
            else:
                out.append({**ex, "group": groups[r["source"]]})
    return out, too_long


def as_dataset(examples: list[dict]):
    from datasets import Dataset

    return Dataset.from_list([{"input_ids": e["input_ids"], "labels": e["labels"]} for e in examples])


def run(cfg: Config) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(cfg.output_dir)
    mix_repo = hub.resolve(cfg.mix_repo)
    train_rows = load_dataset(mix_repo, split="train", token=hf_token())
    val_rows = load_dataset(mix_repo, split="validation", token=hf_token())
    groups = list(dict.fromkeys(cfg.groups.values()))

    tok = load_tokenizer(cfg.model)
    eot = end_of_turn(tok)
    tok.pad_token = choose_pad(tok, eot)
    eot_id, pad_id, eos_id = tok.convert_tokens_to_ids(eot), tok.pad_token_id, tok.eos_token_id
    stop_ids = sorted({eot_id, eos_id})
    if pad_id in (eot_id, eos_id):
        raise ValueError("the pad token must differ from EOS and the end-of-turn token")

    train_ex, train_long = to_examples(train_rows, tok, eot, cfg.max_seq_len, cfg.groups)
    val_ex, val_long = to_examples(val_rows, tok, eot, cfg.max_seq_len, cfg.groups)
    if not all(e["labels"][-1] == eot_id for e in train_ex):
        raise ValueError("an example does not end with a trained end-of-turn token")
    log.info(
        "train %d rows -> %d examples, validation %d -> %d, too long: %d / %d",
        len(train_rows),
        len(train_ex),
        len(val_rows),
        len(val_ex),
        train_long,
        val_long,
    )
    train_ds = as_dataset(train_ex)
    eval_sets = {"all": as_dataset(val_ex), **{g: as_dataset([e for e in val_ex if e["group"] == g]) for g in groups}}

    quantize = cfg.mode == "qlora" or (cfg.mode == "auto" and not bf16_fits(cfg.model))
    log.info("training mode: %s", "QLoRA" if quantize else "bf16 LoRA")
    model = load_model(cfg.model, quantize=quantize)
    model.config.get_text_config().pad_token_id = pad_id
    model.generation_config.pad_token_id = pad_id
    model.generation_config.eos_token_id = stop_ids
    peft_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=lora_targets(model),
        task_type="CAUSAL_LM",
    )

    repo_id = hub.create_private(cfg.model_repo, repo_type="model")
    steps_per_epoch = math.ceil(len(train_ds) / (cfg.batch_size * cfg.grad_accum))
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        seed=cfg.seed,
        data_seed=cfg.seed,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        per_device_eval_batch_size=8,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(cfg.warmup_frac * steps_per_epoch * cfg.epochs)),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        metric_for_best_model="eval_all_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        train_sampling_strategy="group_by_length",
        max_length=cfg.max_seq_len,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    label = f"{cfg.model}, {'QLoRA' if quantize else 'bf16 LoRA'}"
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_sets,
        processing_class=tok,
        peft_config=peft_config,
        callbacks=[Progress(repo_id, groups, tok, stop_ids, pad_id, label), HubCheckpoints(repo_id)],
    )

    t0 = time.time()
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint(output_dir, repo_id))
    except Exception as exc:
        telegram(f"training crashed: {type(exc).__name__}: {exc}")
        raise
    minutes = round((time.time() - t0) / 60, 1)

    best_step = trainer.state.best_global_step
    if best_step and best_step != trainer.state.global_step:
        best = hub.download(repo_id, "best-checkpoint/adapter_model.safetensors", repo_type="model", force=True)
        set_peft_model_state_dict(trainer.model, load_file(str(best)))
        log.info("loaded the best adapter: step %d, eval_all_loss %.4f", best_step, trainer.state.best_metric)

    losses = None
    if cfg.final_eval:
        final = trainer.evaluate()
        losses = {name: final[f"eval_{name}_loss"] for name in ["all", *groups]}
        log.info("final validation loss: %s", {k: round(v, 4) for k, v in losses.items()})

    adapter_dir = output_dir / "best_adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    hub.upload_folder(repo_id, adapter_dir, "adapter", "best LoRA adapter")

    counts = {
        split: Counter(cfg.groups[s] for s in rows["source"])
        for split, rows in (("train", train_rows), ("validation", val_rows))
    }
    card_dir = output_dir / ("merged" if cfg.push_merged else "card")
    card_dir.mkdir(parents=True, exist_ok=True)
    if cfg.push_merged:
        trainer = model = None
        gc.collect()
        torch.cuda.empty_cache()
        merged = PeftModel.from_pretrained(load_model(cfg.model, quantize=False), str(adapter_dir)).merge_and_unload()
        merged.config.get_text_config().pad_token_id = pad_id
        merged.generation_config.pad_token_id = pad_id
        merged.generation_config.eos_token_id = stop_ids
        merged.save_pretrained(card_dir)
        tok.save_pretrained(card_dir)
    card = model_card(cfg, repo_id, mix_repo, eot, quantize, cfg.push_merged, counts, losses, minutes)
    (card_dir / "README.md").write_text(card, encoding="utf-8")
    hub.upload_folder(repo_id, card_dir, "", "model card" + (" and merged bf16 model" if cfg.push_merged else ""))
    if not cfg.push_merged:
        shutil.rmtree(card_dir, ignore_errors=True)
    telegram(f"training done in {minutes} min, best step {best_step}. https://huggingface.co/{repo_id}")
