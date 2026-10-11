from __future__ import annotations

import dataclasses
import typing
from pathlib import Path

import yaml


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        node = data
        *parents, leaf = key.split(".")
        for name in parents:
            node = node.setdefault(name, {})
        node[leaf] = yaml.safe_load(raw)
    return data


def build(cls, data: dict):
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - fields)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {', '.join(unknown)}")
    hints = typing.get_type_hints(cls)
    values = {}
    for name, value in data.items():
        if dataclasses.is_dataclass(hints[name]) and isinstance(value, dict):
            value = build(hints[name], value)
        values[name] = value
    return cls(**values)


def load(cls, path: str | Path, overrides: list[str] | None = None):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return build(cls, apply_overrides(data, overrides or []))
