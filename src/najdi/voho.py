from __future__ import annotations

from najdi.env import hf_token


def load_dialogues(dataset_id: str) -> list[dict]:
    from datasets import load_dataset

    return [dict(r) for r in load_dataset(dataset_id, split="train", token=hf_token())]
