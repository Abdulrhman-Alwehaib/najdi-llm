from __future__ import annotations

import logging
import math
import time
from collections import Counter
from pathlib import Path

from najdi import hub, jsonl
from najdi.llm.dialect import DialectClassifier
from najdi.llm.engine import Engine
from najdi.llm.output import parse_turns, strip_thinking
from najdi.notify import telegram
from najdi.science import assets
from najdi.science.card import FILTERS, dataset_card
from najdi.science.checks import answer_in_reply, answers_match, cheap_filter, final_answer
from najdi.science.config import Config
from najdi.science.tasks import TaskFactory
from najdi.science.wiki import KeywordIndex, PassagePool, keyword_pass
from najdi.text.dedup import is_near_dup, shingles
from najdi.text.dialect import check_per_message
from najdi.text.safety import sensitive_hits
from najdi.voho import load_dialogues

log = logging.getLogger(__name__)


class Generator:
    def __init__(self, cfg: Config, engine: Engine, classifier: DialectClassifier, factory: TaskFactory):
        self.cfg = cfg
        self.engine = engine
        self.classifier = classifier
        self.factory = factory
        self.extra_tokens = cfg.thinking_extra_tokens if cfg.thinking != "off" else 0

    def _row(self, t: dict, msgs: list[dict]) -> dict:
        return {
            "id": t["id"],
            "set": t["set"],
            "topic": t["topic"],
            "source_title": t.get("source_title", ""),
            "source_url": t.get("source_url", ""),
            "generator": self.cfg.engine.model,
            "messages": msgs,
        }

    def batch(self, tasks: list[dict], seen: list[set[str]]) -> tuple[list[dict], list[dict]]:
        cfg, f = self.cfg, self.cfg.filters
        convs = [self.factory.messages(t) for t in tasks]
        params = [self.engine.json_sampling(self.factory.schema(t), cfg.sampling, self.extra_tokens) for t in tasks]
        outs = self.engine.chat(convs, params, thinking=cfg.thinking)

        recs = []
        for t, (text, finish) in zip(tasks, outs):
            rec = {"task_id": t["id"], "set": t["set"], "topic": t["topic"], "status": "dropped", "reason": None}
            answer_text = strip_thinking(text)
            parsed = parse_turns(answer_text, (2, 4))
            answer = parsed[1].get("final_answer", "") if parsed else ""
            answer = answer.strip() if isinstance(answer, str) else ""
            if parsed is None or (t["set"] == "math_logic" and not answer):
                rec.update(reason="format", finish=finish, raw=answer_text[:4000])
            else:
                msgs = parsed[0]
                ok, saudi, fusha = check_per_message(
                    msgs, f.min_saudi_per_100w, f.min_saudi_per_reply, f.max_fusha_ratio
                )
                reason = cheap_filter(msgs, f) or (None if ok else "najdi_words")
                rec.update(messages=msgs, final_answer=answer, saudi_hits=saudi, fusha_hits=fusha, reason=reason)
                if reason == "sensitive":
                    rec["sensitive_terms"] = sorted(sensitive_hits("\n".join(m["content"] for m in msgs)))
            recs.append(rec)

        alive = [r for r in recs if r["reason"] is None]
        replies = [m["content"] for r in alive for m in r["messages"] if m["role"] == "assistant"]
        labels = iter(self.classifier.labels(replies) if replies else [])
        for r in alive:
            r["dialect"] = [next(labels) for m in r["messages"] if m["role"] == "assistant"]
            if any(label != "GLF" for label in r["dialect"]):
                r["reason"] = "dialect_clf"

        batch_openers = []
        for r in recs:
            if r["reason"] is None:
                sh = shingles(r["messages"][0]["content"])
                if is_near_dup(sh, seen, f.dedup_jaccard) or is_near_dup(sh, batch_openers, f.dedup_jaccard):
                    r["reason"] = "dedup"
                else:
                    batch_openers.append(sh)

        self._math_check(recs)

        rows = []
        for t, r in zip(tasks, recs):
            if r["reason"] is None:
                r["status"] = "kept"
                rows.append(self._row(t, r["messages"]))
                seen.append(shingles(r["messages"][0]["content"]))
        return rows, recs

    def _math_check(self, recs: list[dict]) -> None:
        todo = []
        for r in recs:
            if r["reason"] is None and r["set"] == "math_logic":
                if self.cfg.filters.check_answer_in_reply and not answer_in_reply(
                    r["final_answer"], r["messages"][1]["content"]
                ):
                    r.update(reason="math_check", math_note="answer not in first reply")
                else:
                    todo.append(r)
        if not todo:
            return
        convs = [
            [{"role": "user", "content": assets.VERIFY_PROMPT.format(problem=r["messages"][0]["content"])}]
            for r in todo
        ]
        outs = self.engine.chat(convs, self.engine.sampling(**self.cfg.verify_sampling), thinking="full")
        unsure = []
        for r, (text, finish) in zip(todo, outs):
            r["verify_answer"] = final_answer(text, finish)
            if r["verify_answer"] is None:
                r.update(reason="math_check", math_note="re-solve gave no final answer")
            elif not answers_match(r["final_answer"], r["verify_answer"]):
                unsure.append(r)
        if not unsure:
            return
        convs = [
            [
                {
                    "role": "user",
                    "content": assets.ANSWER_JUDGE_PROMPT.format(
                        problem=r["messages"][0]["content"], a=r["final_answer"], b=r["verify_answer"]
                    ),
                }
            ]
            for r in unsure
        ]
        outs = self.engine.chat(convs, self.engine.choice_sampling(["yes", "no"]))
        for r, (text, _) in zip(unsure, outs):
            r["judge"] = text.strip()
            if r["judge"] != "yes":
                r.update(reason="math_check", math_note="answers differ")


def plan_batch(need: dict[str, int], attempts: list[dict], batch_tasks: int) -> dict[str, int]:
    want = {}
    for s, n in need.items():
        tried = [a for a in attempts if a["set"] == s]
        rate = (sum(a["status"] == "kept" for a in tried) + 1) / (len(tried) + 2)
        want[s] = math.ceil(n / max(rate, 0.1)) if n else 0
    scale = min(1.0, batch_tasks / max(sum(want.values()), 1))
    return {s: math.ceil(w * scale) for s, w in want.items()}


def filter_report(recs: list[dict]) -> dict[str, int]:
    removed = Counter(r["reason"] for r in recs if r["status"] == "dropped")
    left = len(recs)
    for name in FILTERS:
        k = removed.get(name, 0)
        log.info("%-15s passed %5d of %-5d (removed %d)", name, left - k, left, k)
        left -= k
    log.info("kept %d of %d", sum(r["status"] == "kept" for r in recs), len(recs))
    return {name: removed.get(name, 0) for name in FILTERS}


def push(cfg: Config, data_path: Path, kept: list[dict], attempts: list[dict], message: str) -> bool:
    try:
        repo_id = hub.push(
            cfg.repo_id,
            {"data/train.jsonl": data_path, "README.md": dataset_card(cfg, kept, attempts)},
            message,
        )
    except Exception as exc:
        log.warning("hub push failed: %s", exc)
        return False
    log.info("pushed %d rows to %s", len(kept), repo_id)
    return True


def run(cfg: Config) -> None:
    root = Path(cfg.output_dir)
    out_dir = root / cfg.engine.model.split("/")[-1] / f"thinking-{cfg.thinking}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / ("pilot.jsonl" if cfg.pilot else "data.jsonl")
    attempts_path = out_dir / ("pilot_attempts.jsonl" if cfg.pilot else "attempts.jsonl")

    everyday = [r["messages"] for r in load_dialogues(cfg.voho_id) if r["set"] == "everyday"]
    anchors = [msgs for msgs in everyday if not sensitive_hits(" ".join(m["content"] for m in msgs))]
    index = KeywordIndex(cfg.domain.keywords)
    candidates = keyword_pass(cfg.wiki, index, cfg.seed, root / "wiki_candidates.jsonl")
    log.info("%d candidates by keyword topic: %s", len(candidates), dict(Counter(c["kw_topic"] for c in candidates)))

    engine = Engine(cfg.engine, cfg.seed)
    classifier = DialectClassifier(cfg.filters.dialect_classifier)
    pool = PassagePool(candidates, cfg.domain, engine, root / "wiki_labels.jsonl", cfg.wiki.label_chunk)
    factory = TaskFactory(cfg, anchors)
    generator = Generator(cfg, engine, classifier, factory)

    if cfg.pilot:
        for p in (data_path, attempts_path):
            p.unlink(missing_ok=True)
    for p in (data_path, attempts_path):
        jsonl.repair(p)
    kept, attempts = jsonl.read(data_path), jsonl.read(attempts_path)
    done = {r["id"] for r in kept} | {a["task_id"] for a in attempts}
    prefix = f"{cfg.id_prefix}-"
    used = {d[len(prefix) :] for d in done if d.startswith(prefix)}
    next_math = 1 + max((int(d[len("math-") :]) for d in done if d.startswith("math-")), default=-1)
    seen = [shingles(r["messages"][0]["content"]) for r in kept]
    target = cfg.pilot_rows if cfg.pilot else cfg.target_rows
    goal = {"science": round(target * cfg.science_frac)}
    goal["math_logic"] = target - goal["science"]
    pushed_bucket, pushed_rows = len(kept) // cfg.push_every, None
    told_bucket = len(kept) // cfg.telegram_every
    name = f"{cfg.domain.name} {'pilot' if cfg.pilot else 'run'} ({cfg.engine.model.split('/')[-1]})"
    log.info("%s -> %s | goal %s | resuming with %d kept, %d attempts", name, data_path, goal, len(kept), len(attempts))
    telegram(f"{name} started: goal {target} rows, {len(kept)} already kept.")

    t0 = time.time()
    try:
        while True:
            have = Counter(r["set"] for r in kept)
            need = {s: max(0, goal[s] - have[s]) for s in goal}
            if cfg.pilot:
                if attempts:
                    break
                plan = dict(goal)
            elif not any(need.values()):
                break
            elif len(attempts) >= 4 * target:
                log.warning("stopping: 4x the target has been attempted")
                break
            else:
                plan = plan_batch(need, attempts, cfg.batch_tasks)
            topic_counts = Counter(r["topic"] for r in kept if r["set"] == "science")
            passages = pool.take(plan["science"], used, topic_counts) if plan["science"] else []
            tasks = [factory.science(p) for p in passages]
            tasks += [factory.math(next_math + i) for i in range(plan["math_logic"])]
            next_math += plan["math_logic"]
            if not tasks:
                log.warning("nothing left to generate")
                break
            rows, recs = generator.batch(tasks, seen)
            jsonl.append(data_path, rows)
            jsonl.append(attempts_path, recs)
            kept += rows
            attempts += recs
            used.update(t["passage_id"] for t in tasks if t["set"] == "science")
            have = Counter(r["set"] for r in kept)
            minutes = (time.time() - t0) / 60
            log.info(
                "[%.0f min] batch kept %d/%d | science %d/%d, math_logic %d/%d",
                minutes,
                len(rows),
                len(tasks),
                have["science"],
                goal["science"],
                have["math_logic"],
                goal["math_logic"],
            )
            if (
                not cfg.pilot
                and len(kept) // cfg.push_every > pushed_bucket
                and push(cfg, data_path, kept, attempts, f"checkpoint: {len(kept)} rows")
            ):
                pushed_bucket, pushed_rows = len(kept) // cfg.push_every, len(kept)
            if len(kept) // cfg.telegram_every > told_bucket:
                told_bucket = len(kept) // cfg.telegram_every
                telegram(f"{name}: {len(kept)}/{target} rows kept from {len(attempts)} generated, {minutes:.0f} min.")
    except Exception as exc:
        telegram(f"{name} crashed with {len(kept)} rows kept: {type(exc).__name__}: {exc}")
        raise

    removed = filter_report(attempts)
    if not cfg.pilot and pushed_rows != len(kept):
        push(cfg, data_path, kept, attempts, f"final: {len(kept)} rows")
    telegram(
        f"{name} finished: {len(kept)} rows kept from {len(attempts)} generated in {(time.time() - t0) / 60:.0f} min. "
        + ", ".join(f"{k} {v}" for k, v in removed.items())
    )
