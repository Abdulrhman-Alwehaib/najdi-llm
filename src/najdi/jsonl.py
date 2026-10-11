from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterable
from pathlib import Path


def read(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            with contextlib.suppress(json.JSONDecodeError):
                rows.append(json.loads(line))
    return rows


def append(path: str | Path, rows: Iterable[dict]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write(path: str | Path, rows: Iterable[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def repair(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            path.write_bytes(data[: data.rfind(b"\n") + 1])
