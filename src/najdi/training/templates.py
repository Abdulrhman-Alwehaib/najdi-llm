from __future__ import annotations

import re

from najdi.env import hf_token

INSTRUCT_TEMPLATE = {
    "Qwen/Qwen3.5-9B-Base": "Qwen/Qwen3.5-9B",
    "google/gemma-4-E4B": "google/gemma-4-E4B-it",
    "mistralai/Ministral-3-8B-Base-2512": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "QCRI/Fanar-1-9B": "QCRI/Fanar-1-9B-Instruct",
}
TEMPLATE_KWARGS = {"enable_thinking": False}
_DEFAULT_SYSTEM = re.compile(r"\{%-?\s*set default_system_message = '(?:[^'\\]|\\.)*'\s*-?%\}", re.S)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    kwargs = {"token": hf_token()}
    if "mistral" in model_id.lower():
        kwargs["fix_mistral_regex"] = True
    tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
    template_id = INSTRUCT_TEMPLATE.get(model_id, model_id)
    if template_id != model_id:
        tok.chat_template = AutoTokenizer.from_pretrained(template_id, **kwargs).chat_template
    if not tok.chat_template:
        raise ValueError(f"{model_id} has no chat template")
    tok.chat_template = _DEFAULT_SYSTEM.sub("{%- set default_system_message = '' %}", tok.chat_template)
    return tok


def end_of_turn(tok) -> str:
    marker = "QZXQ"
    text = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": marker}], tokenize=False, **TEMPLATE_KWARGS
    )
    tail = text[text.rindex(marker) + len(marker) :]
    specials = set(tok.all_special_tokens) | set(tok.get_added_vocab())
    for token in sorted(specials, key=len, reverse=True):
        if token and tail.startswith(token):
            return token
    raise ValueError(f"no special token after the assistant message: {tail[:40]!r}")


def choose_pad(tok, eot: str) -> str:
    bad = {tok.convert_tokens_to_ids(eot), tok.eos_token_id}
    vocab = tok.get_vocab()
    for cand in (tok.pad_token, "<pad>", "<|pad|>", "<|vision_pad|>", "<|fim_pad|>", "<unk>", "<|endoftext|>"):
        if cand and cand in vocab and vocab[cand] not in bad:
            return cand
    raise ValueError("no pad token that differs from EOS and the end-of-turn token")


def turn_pairs(tok, messages: list[dict]) -> list[tuple[str, str]]:
    return [
        (
            tok.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS),
            m["content"],
        )
        for i, m in enumerate(messages)
        if m["role"] == "assistant"
    ]


def _encode(tok, pieces: list[tuple[str, str]], eot_id: int, score_eot: bool) -> dict:
    ids, labels = [], []
    for context, reply in pieces:
        c = tok(context, add_special_tokens=False)["input_ids"]
        r = tok(reply, add_special_tokens=False)["input_ids"]
        ids += c + r + [eot_id]
        labels += [-100] * len(c) + r + ([eot_id] if score_eot else [-100])
    return {"input_ids": ids, "labels": labels}


def build_examples(tok, messages: list[dict], eot: str, score_eot: bool = True) -> list[dict]:
    eot_id = tok.convert_tokens_to_ids(eot)
    pairs = turn_pairs(tok, messages)
    pieces, so_far = [], ""
    for prompt, reply in pairs:
        if not prompt.startswith(so_far):
            return [_encode(tok, [pair], eot_id, score_eot) for pair in pairs]
        pieces.append((prompt[len(so_far) :], reply))
        so_far = prompt + reply + eot
    return [_encode(tok, pieces, eot_id, score_eot)]
