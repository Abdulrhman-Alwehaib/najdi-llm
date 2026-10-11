import json

import pytest


class FakeTokenizer:
    specials = ["<s>", "<|im_start|>", "<|im_end|>", "<pad>"]

    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(self.specials)}
        self.pad_token = None
        self.eos_token_id = self.vocab["<s>"]
        self.all_special_tokens = list(self.specials)

    def get_added_vocab(self):
        return {}

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for token in self.specials:
                if text.startswith(token, i):
                    ids.append(self.vocab[token])
                    i += len(token)
                    break
            else:
                ids.append(100 + ord(text[i]))
                i += 1
        return {"input_ids": ids}


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def places_file(tmp_path):
    items = [
        {
            "name": "روضة خريم",
            "region": "الرياض",
            "title": "روضة خريم",
            "aliases": [],
            "summary": "روضة خضراء شمال شرق الرياض.",
            "url": "https://ar.wikipedia.org/wiki/x",
        },
        {
            "name": "الثمامة",
            "region": "الرياض",
            "title": "الثمامة",
            "aliases": ["منتزه الثمامة"],
            "summary": "منطقة صحراوية شمال الرياض.",
            "url": "https://ar.wikipedia.org/wiki/y",
        },
        {
            "name": "العلا",
            "region": "الغربية",
            "title": "العلا",
            "aliases": [],
            "summary": "مدينة تاريخية في المدينة المنورة.",
            "url": "https://ar.wikipedia.org/wiki/z",
        },
        {
            "name": "أبها",
            "region": "الجنوب",
            "title": "أبها",
            "aliases": [],
            "summary": "مدينة جبلية في عسير.",
            "url": "https://ar.wikipedia.org/wiki/w",
        },
    ]
    path = tmp_path / "places.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def anchors():
    chat = [{"role": "user", "content": "هلا وش عندك"}, {"role": "assistant", "content": "ابشر وش تبي"}]
    return {"enterprise": [chat] * 3, "everyday": [chat] * 3}
