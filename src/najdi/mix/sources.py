from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter

from najdi import hub
from najdi.env import hf_token
from najdi.mix.config import Config, Report
from najdi.text.arabic import normalize, phrase_hits, phrases, word_count
from najdi.text.safety import sensitive_hits

log = logging.getLogger(__name__)

_TRANSLATION = re.compile(r"ترجم|translat", re.I)
_ENGLISH_MENTION = re.compile(r"الإنجليزي|الانجليزي|الإنكليزي|الانكليزي|English", re.I)
_ATTACHMENT = re.compile(r"مرفق|الرابط التالي|الصورة التالية|الفيديو التالي")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_ABSHIR = re.compile(r"^\s*[اأإآ]بشر(?:ي|وا)?(?![^\W\d_])[\s،,.!؟?:؛…-]*")
_ABSHIR_PHRASE = re.compile(r"(?:بعزك|بعزكم|بالخير|بسعدك|بالفرج)(?![^\W\d_])")


def chat_text(r: dict) -> str:
    return "\n".join(m["content"] for m in r["messages"] if m["role"] != "system")


def replies(r: dict) -> str:
    return "\n".join(m["content"] for m in r["messages"] if m["role"] == "assistant")


def load(cfg: Config, report: Report) -> list[dict]:
    from datasets import load_dataset

    token = hf_token()
    rows = []
    chat_repo = hub.resolve(cfg.chat_repo)
    for subset in ("helpful", "identity"):
        chat = load_dataset(chat_repo, subset, split="train", token=token)
        rows += [{"id": r["id"], "source": subset, "messages": r["messages"], "_strata": r["topic"]} for r in chat]
        log.info("%s [%s]: %d rows", chat_repo, subset, len(chat))
    for name in cfg.science_repos:
        science = load_dataset(hub.resolve(name), split="train", token=token)
        rows += [{"id": r["id"], "source": r["set"], "messages": r["messages"], "_strata": r["topic"]} for r in science]
        log.info("%s: %d rows %s", name, len(science), dict(Counter(r["set"] for r in science)))
    voho = load_dataset(hub.resolve(cfg.voho_id), split="train", token=token)
    report.voho_system_rows = sum(any(m["role"] == "system" for m in r["messages"]) for r in voho)
    rows += [
        {
            "id": r["id"],
            "source": f"voho_{r['set']}",
            "messages": r["messages"],
            "_strata": f"{r['topic']} / {r['scenario']}",
        }
        for r in voho
    ]
    if cfg.fusha_source == "cidar":
        ds = load_dataset("arbml/CIDAR", split="train")
        pairs = [(f"cidar-{i:05d}", r["instruction"], r["output"]) for i, r in enumerate(ds)]
    elif cfg.fusha_source == "aya":
        ds = load_dataset("CohereLabs/aya_dataset", split="train")
        keep = [i for i, code in enumerate(ds["language_code"]) if code == "arb"]
        pairs = [(f"aya-{i:06d}", r["inputs"], r["targets"]) for i, r in zip(keep, ds.select(keep))]
    else:
        raise ValueError(f"fusha_source must be cidar or aya, not {cfg.fusha_source!r}")
    for pid, instruction, output in pairs:
        rows.append(
            {
                "id": pid,
                "source": "fusha",
                "_strata": "",
                "messages": [
                    {"role": "system", "content": cfg.fusha_system_prompt},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ],
            }
        )
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("ids must be unique across sources")
    report.loaded = Counter(r["source"] for r in rows)
    return rows


class Cleaner:
    def __init__(self, cfg: Config, report: Report):
        self.cfg = cfg
        self.report = report
        self.non_najdi = (
            re.compile(r"(?<![^\W\d_])(?:" + "|".join(map(normalize, cfg.non_najdi_words)) + r")(?![^\W\d_])")
            if cfg.non_najdi_words
            else None
        )
        self.intimate = phrases(cfg.voho_intimate)
        self.family_role = phrases(cfg.voho_family_role)
        self.personal = phrases(cfg.voho_personal)

    def fix_roles(self, r: dict) -> bool:
        src, chat, fixes = r["source"], [], set()
        for m in r["messages"]:
            role, content = m.get("role"), m.get("content")
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                return False
            text = content.strip()
            if text != content:
                fixes.add("trimmed")
            if not text:
                fixes.add("empty_turn_dropped")
                continue
            if role == "system":
                if src != "fusha":
                    fixes.add("system_removed")
                continue
            if chat and chat[-1]["role"] == role:
                chat[-1]["content"] += "\n\n" + text
                fixes.add("same_speaker_merged")
            else:
                chat.append({"role": role, "content": text})
        while chat and chat[0]["role"] != "user":
            chat.pop(0)
            fixes.add("leading_assistant_dropped")
        while chat and chat[-1]["role"] != "assistant":
            chat.pop()
            fixes.add("trailing_user_dropped")
        if len(chat) < 2:
            return False
        if src == "fusha":
            chat.insert(0, {"role": "system", "content": self.cfg.fusha_system_prompt})
        r["messages"] = chat
        for f in fixes:
            self.report.fixed[f][src] += 1
        return True

    def fusha_problem(self, r: dict) -> str | None:
        cfg = self.cfg
        instruction, output = r["messages"][1]["content"], r["messages"][-1]["content"]
        if word_count(output) < cfg.fusha_min_output_words or word_count(instruction) < cfg.fusha_min_instruction_words:
            return "fusha_short"
        if _TRANSLATION.search(instruction):
            return "fusha_translation"
        both = instruction + "\n" + output
        if (
            len(_LATIN_WORD.findall(both)) > cfg.fusha_max_latin_words
            or _ENGLISH_MENTION.search(both)
            or _ATTACHMENT.search(instruction)
        ):
            return "fusha_external"
        return None

    def voho_problem(self, r: dict) -> str | None:
        text = replies(r)
        if phrase_hits(text, self.intimate):
            return "voho_intimate"
        if phrase_hits(text, self.family_role) or phrase_hits(text, self.personal, claims=True):
            return "voho_personal"
        return None

    def cut_abshir(self, rows: list[dict]) -> None:
        found = []
        for r in rows:
            if r["source"] not in ("helpful", "identity"):
                continue
            for i, m in enumerate(r["messages"]):
                hit = _ABSHIR.match(m["content"]) if m["role"] == "assistant" else None
                rest = m["content"][hit.end() :] if hit else ""
                if hit and not _ABSHIR_PHRASE.match(rest) and len(rest.split()) >= 3:
                    found.append((r, i, hit.end()))
        found.sort(key=lambda f: hashlib.md5(f"{self.cfg.seed}:{f[0]['id']}:{f[1]}".encode()).hexdigest())
        cut = found[: round(self.cfg.abshir_cut_frac * len(found))]
        for r, i, end in cut:
            r["messages"][i] = {**r["messages"][i], "content": r["messages"][i]["content"][end:]}
        self.report.abshir = {
            s: (sum(f[0]["source"] == s for f in found), sum(c[0]["source"] == s for c in cut))
            for s in ("helpful", "identity")
        }

    def __call__(self, rows: list[dict]) -> list[dict]:
        report = self.report
        rows = report.drop_where(rows, "broken_roles", lambda r: not self.fix_roles(r))
        kept = []
        for r in rows:
            problem = self.fusha_problem(r) if r["source"] == "fusha" else None
            if problem:
                report.removed[problem]["fusha"] += 1
            else:
                kept.append(r)
        rows = kept
        if self.cfg.screen_sensitive:
            rows = report.drop_where(rows, "sensitive", lambda r: sensitive_hits(chat_text(r)))
        if self.non_najdi:
            rows = report.drop_where(
                rows,
                "non_najdi",
                lambda r: r["source"] in ("helpful", "identity") and self.non_najdi.search(normalize(replies(r))),
            )
        kept = []
        for r in rows:
            problem = self.voho_problem(r) if r["source"].startswith("voho_") else None
            if problem:
                report.removed[problem][r["source"]] += 1
            else:
                kept.append(r)
        self.cut_abshir(kept)
        return kept
