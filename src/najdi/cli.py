from __future__ import annotations

import argparse
import importlib
import logging

from najdi import settings
from najdi.env import load_dotenv

COMMANDS = {
    "science": ("najdi.science.config:Config", "najdi.science.pipeline:run"),
    "chat-data": ("najdi.chatdata.config:Config", "najdi.chatdata.pipeline:run"),
    "clean": ("najdi.cleaning:Config", "najdi.cleaning:run"),
    "mix": ("najdi.mix.config:Config", "najdi.mix.pipeline:run"),
    "compare": ("najdi.training.config:CompareConfig", "najdi.training.compare:run"),
    "train": ("najdi.training.config:Config", "najdi.training.train:run"),
    "serve": ("najdi.serving.app:Config", "najdi.serving.app:run"),
}


def _resolve(path: str):
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="najdi")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = sub.add_parser(name)
        command.add_argument("-c", "--config", required=True)
        command.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    load_dotenv()
    config_cls, run = COMMANDS[args.command]
    cfg = settings.load(_resolve(config_cls), args.config, args.overrides)
    _resolve(run)(cfg)
