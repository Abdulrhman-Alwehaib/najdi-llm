from __future__ import annotations

import logging
from dataclasses import dataclass

from najdi.notify import telegram

log = logging.getLogger(__name__)


@dataclass
class Config:
    model_repo: str = ""
    adapter: str = "best-checkpoint"
    system_prompt: str = ""
    fusha_system: str = "رد بالفصحى"
    temperature: float = 0.7
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1
    share: bool = False
    host: str = "0.0.0.0"
    port: int = 7860
    username: str = ""
    password: str = ""


def to_messages(message: str, history: list[dict], fusha: bool, cfg: Config) -> list[dict]:
    system = cfg.fusha_system if fusha else cfg.system_prompt
    msgs = [{"role": "system", "content": system}] if system else []
    for turn in history:
        if (
            turn.get("role") in ("user", "assistant")
            and isinstance(turn.get("content"), str)
            and turn["content"].strip()
        ):
            msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append({"role": "user", "content": message})
    return msgs


def build_app(model, cfg: Config):
    import gradio as gr

    def reply(message, history, fusha, temperature, max_new_tokens):
        yield from model.stream(
            to_messages(message, history, fusha, cfg), temperature, max_new_tokens, cfg.repetition_penalty
        )

    return gr.ChatInterface(
        reply,
        chatbot=gr.Chatbot(rtl=True, height=560, label="Najdi chat"),
        additional_inputs=[
            gr.Checkbox(value=False, label=f"Fusha ({cfg.fusha_system})"),
            gr.Slider(0, 1.5, value=cfg.temperature, step=0.05, label="temperature"),
            gr.Slider(64, 1024, value=cfg.max_new_tokens, step=32, label="max new tokens"),
        ],
        additional_inputs_accordion=gr.Accordion("Settings", open=False),
        examples=[
            ["هلا والله، وش أخبارك؟", False, cfg.temperature, cfg.max_new_tokens],
            ["ليش السما لونها أزرق؟", False, cfg.temperature, cfg.max_new_tokens],
            ["ما أهمية القراءة في حياة الإنسان؟", True, cfg.temperature, cfg.max_new_tokens],
        ],
        title="Najdi chat",
        description=f"`{model.repo}` on `{model.base_id}`",
        fill_height=True,
    )


def run(cfg: Config) -> None:
    from najdi.serving.model import ChatModel

    model = ChatModel(cfg.model_repo, cfg.adapter)
    app = build_app(model, cfg)
    auth = (cfg.username, cfg.password) if cfg.username and cfg.password else None
    _, local_url, share_url = app.launch(
        server_name=cfg.host, server_port=cfg.port, share=cfg.share, auth=auth, prevent_thread_lock=True, inline=False
    )
    log.info("chat page: %s", share_url or local_url)
    if share_url:
        telegram(f"Najdi chat is up: {share_url}")
    app.block_thread()
