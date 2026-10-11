from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(start: Path | None = None) -> Path | None:
    folder = (start or Path.cwd()).resolve()
    for candidate in [folder, *folder.parents][:3]:
        path = candidate / ".env"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and key and not key.startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip())
            return path
    return None


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")
