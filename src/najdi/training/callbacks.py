from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from transformers import TrainerCallback

from najdi import hub
from najdi.env import hf_token
from najdi.notify import telegram
from najdi.training.prompts import FIXED_PROMPTS
from najdi.training.templates import TEMPLATE_KWARGS

log = logging.getLogger(__name__)


def answer_fixed_prompts(model, tok, stop_ids: list[int], pad_id: int) -> list[dict]:
    import torch

    was_training = model.training
    model.eval()
    answers = []
    with torch.no_grad():
        for kind, msgs in FIXED_PROMPTS:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS)
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
            out = model.generate(
                **ids,
                max_new_tokens=300,
                do_sample=False,
                repetition_penalty=1.05,
                use_cache=True,
                eos_token_id=stop_ids,
                pad_token_id=pad_id,
            )
            answers.append(
                {
                    "kind": kind,
                    "prompt": msgs[-1]["content"],
                    "system": msgs[0]["content"] if len(msgs) > 1 else "",
                    "answer": tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True).strip(),
                }
            )
    if was_training:
        model.train()
    return answers


class Progress(TrainerCallback):
    def __init__(self, repo_id: str, groups: list[str], tok, stop_ids: list[int], pad_id: int, label: str):
        self.repo_id = repo_id
        self.groups = groups
        self.tok = tok
        self.stop_ids = stop_ids
        self.pad_id = pad_id
        self.label = label
        self.round: dict[str, float] = {}

    def on_train_begin(self, args, state, control, **kwargs):
        telegram(f"training started: {self.label}, {state.max_steps} steps, resuming at step {state.global_step}.")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.round.update(
            {k[len("eval_") : -len("_loss")]: v for k, v in (metrics or {}).items() if k.endswith("_loss")}
        )
        names = ["all", *self.groups]
        if all(name in self.round for name in names):
            line = ", ".join(f"{name} {self.round[name]:.3f}" for name in names)
            log.info("[step %d] validation loss: %s", state.global_step, line)
            telegram(f"eval at step {state.global_step}/{state.max_steps} (epoch {state.epoch:.2f}): {line}")
            self.round = {}

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        answers = answer_fixed_prompts(model, self.tok, self.stop_ids, self.pad_id)
        out = Path(args.output_dir)
        (out / f"fixed_prompts_epoch{epoch}.json").write_text(
            json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for a in answers:
            log.info("[%s] %s\n%s", a["kind"], a["prompt"], a["answer"])
        folder = out / f"epoch-{epoch}-adapter"
        model.save_pretrained(folder)
        self.tok.save_pretrained(folder)
        try:
            hub.upload_folder(self.repo_id, folder, f"epoch-{epoch}", f"LoRA adapter after epoch {epoch}")
        except Exception as exc:
            log.warning("epoch adapter upload failed: %s", exc)
        telegram(f"epoch {epoch} done, adapter pushed to {self.repo_id}/epoch-{epoch}.")


class HubCheckpoints(TrainerCallback):
    def __init__(self, repo_id: str):
        self.repo_id = repo_id

    def on_save(self, args, state, control, **kwargs):
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        is_best = state.best_global_step == state.global_step
        try:
            hub.upload_folder(self.repo_id, ckpt, "last-checkpoint", f"checkpoint at step {state.global_step}")
            if is_best:
                hub.upload_folder(
                    self.repo_id,
                    ckpt,
                    "best-checkpoint",
                    f"best so far: step {state.global_step}, eval_all_loss {state.best_metric:.4f}",
                    allow=["adapter_model.safetensors", "adapter_config.json"],
                )
        except Exception as exc:
            log.warning("step %d: hub upload failed, keeping the local checkpoint (%s)", state.global_step, exc)
            return
        for d in Path(args.output_dir).glob("checkpoint-*"):
            shutil.rmtree(d, ignore_errors=True)
        log.info("step %d: checkpoint pushed%s", state.global_step, " (new best)" if is_best else "")


def resume_checkpoint(output_dir: Path, repo_id: str) -> str | None:
    from huggingface_hub import snapshot_download
    from transformers.trainer_utils import get_last_checkpoint

    for d in output_dir.glob("checkpoint-*") if output_dir.exists() else []:
        if not (d / "trainer_state.json").exists():
            shutil.rmtree(d, ignore_errors=True)
    local = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    local_step = int(local.rsplit("-", 1)[1]) if local else -1
    if "last-checkpoint/trainer_state.json" not in hub.api().list_repo_files(repo_id):
        return local
    hub_state = hub.download(repo_id, "last-checkpoint/trainer_state.json", repo_type="model")
    hub_step = json.loads(hub_state.read_text())["global_step"]
    if hub_step <= local_step:
        return local
    staging = output_dir / "_hub"
    shutil.rmtree(staging, ignore_errors=True)
    snapshot_download(repo_id, allow_patterns="last-checkpoint/*", local_dir=staging, token=hf_token())
    dest = output_dir / f"checkpoint-{hub_step}"
    shutil.move(str(staging / "last-checkpoint"), str(dest))
    log.info("resuming from the hub checkpoint at step %d", hub_step)
    return str(dest)
