from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from najdi.chatdata import assets
from najdi.text.arabic import normalize

log = logging.getLogger(__name__)

API = "https://ar.wikipedia.org/w/api.php?"


def _query(titles: list[str]) -> dict[str, dict]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|pageprops",
        "exintro": 1,
        "explaintext": 1,
        "exlimit": 20,
        "redirects": 1,
        "titles": "|".join(titles),
    }
    request = urllib.request.Request(API + urllib.parse.urlencode(params), headers={"User-Agent": "najdi-llm/0.1"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=30) as r:
                data = json.load(r)["query"]
            break
        except Exception as exc:
            if attempt == 4:
                raise
            log.warning("wikipedia request failed (%s), retrying", type(exc).__name__)
            time.sleep(10 * (attempt + 1))
    alias = {x["from"]: x["to"] for x in data.get("normalized", []) + data.get("redirects", [])}
    pages = {p["title"]: p for p in data.get("pages", {}).values()}
    out = {}
    for title in titles:
        page = pages.get(alias.get(alias.get(title, title), alias.get(title, title)))
        if page and "missing" not in page and "disambiguation" not in page.get("pageprops", {}):
            out[title] = page
    return out


def short_summary(text: str, max_words: int) -> str:
    text = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", text.split("\n")[0])
    out = []
    for w in " ".join(text.split()).split():
        out.append(w)
        if len(out) >= max_words or (w.endswith(".") and len(out) >= 20):
            break
    return " ".join(out).rstrip("،,") + ("" if out and out[-1].endswith(".") else "...")


def fetch(summary_words: int) -> list[dict]:
    flat = [(region, *c) for region, cands in assets.PLACE_CANDIDATES.items() for c in cands]
    found = {}
    for i in range(0, len(flat), 20):
        found.update(_query([c[2] for c in flat[i : i + 20]]))
        time.sleep(1)
    places = []
    for region, name, title, aliases in flat:
        page = found.get(title)
        if not page or len(page.get("extract", "").split()) < 8:
            log.warning("no usable article for %s", title)
            continue
        places.append(
            {
                "name": name,
                "region": region,
                "title": page["title"],
                "aliases": aliases,
                "summary": short_summary(page["extract"], summary_words),
                "url": "https://ar.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_")),
            }
        )
    return places


class Places:
    def __init__(self, path: Path, summary_words: int, per_prompt: int):
        if path.exists():
            self.items = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.items = fetch(summary_words)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")
        self.per_prompt = per_prompt
        self.summaries = {p["name"]: p["summary"] for p in self.items}
        self.names = {
            normalize(n) for p in self.items for n in [p["name"], re.sub(r"\s*\(.*\)", "", p["title"]), *p["aliases"]]
        }
        self.phrases = sorted({tuple(n.split()) for n in self.names})

    def for_persona(self, persona: dict, rng: random.Random) -> list[dict]:
        own = [p for p in self.items if p["region"] == persona["region"]]
        other = [p for p in self.items if p["region"] != persona["region"]]
        k = min(len(own), self.per_prompt - 3)
        return rng.sample(own, k) + rng.sample(other, self.per_prompt - k)
