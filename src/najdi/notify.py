from __future__ import annotations

import logging
import os
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)


def telegram(text: str) -> None:
    token, chat_id = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text[:4000]}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=10)
    except Exception as exc:
        log.warning("telegram message failed: %s", type(exc).__name__)
