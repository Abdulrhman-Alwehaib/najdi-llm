from __future__ import annotations


class DialectClassifier:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()

    def labels(self, texts: list[str], batch_size: int = 16) -> list[str]:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(
                    texts[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                ids = self.model(**enc).logits.argmax(-1).tolist()
                out += [self.model.config.id2label[j] for j in ids]
        return out
