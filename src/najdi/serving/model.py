from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Thread

from najdi import hub
from najdi.env import hf_token
from najdi.training.templates import TEMPLATE_KWARGS, end_of_turn


class ChatModel:
    def __init__(self, repo: str, adapter: str, quantize_below_gb: float = 24.0):
        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        self.repo = hub.resolve(repo)
        snapshot = snapshot_download(self.repo, allow_patterns=[f"{adapter}/*"], token=hf_token())
        adapter_dir = Path(snapshot) / adapter
        self.base_id = PeftConfig.from_pretrained(str(adapter_dir)).base_model_name_or_path
        self.tok = AutoTokenizer.from_pretrained(self.base_id, token=hf_token())
        kwargs = {"token": hf_token(), "dtype": torch.bfloat16, "device_map": "auto"}
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        self.quantized = 0 < gpu_gb < quantize_below_gb
        if self.quantized:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        try:
            base = AutoModelForCausalLM.from_pretrained(self.base_id, **kwargs)
        except ValueError:
            base = AutoModelForImageTextToText.from_pretrained(self.base_id, **kwargs)
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
        if not lora_b or not any(bool(p.detach().abs().sum() > 0) for p in lora_b):
            raise RuntimeError("the adapter did not load: its layer names do not match the base model")
        if not self.quantized:
            model = model.merge_and_unload()
        self.model = model.eval()
        self.stop_ids = sorted({self.tok.convert_tokens_to_ids(end_of_turn(self.tok)), self.tok.eos_token_id} - {None})
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop_ids[0]

    def stream(
        self, messages: list[dict], temperature: float, max_new_tokens: int, repetition_penalty: float
    ) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS)
        inputs = self.tok(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        args = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=self.stop_ids,
            pad_token_id=self.pad_id,
            repetition_penalty=repetition_penalty,
        )
        if temperature > 0:
            args.update(do_sample=True, temperature=float(temperature), top_p=0.9)
        else:
            args.update(do_sample=False)
        Thread(target=self.model.generate, kwargs=args, daemon=True).start()
        text = ""
        for piece in streamer:
            text += piece
            yield text
