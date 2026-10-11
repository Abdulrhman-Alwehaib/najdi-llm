from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from najdi import hub, jsonl
from najdi.chatdata.card import coverage, dataset_card, filter_table
from najdi.chatdata.config import Config
from najdi.chatdata.filters import INTIMATE, RuleFilter, dedup_text, row_openers
from najdi.chatdata.judge import Judge
from najdi.chatdata.places import Places
from najdi.chatdata.prompts import PromptBuilder
from najdi.chatdata.seeds import SeedBuilder
from najdi.llm.dialect import DialectClassifier
from najdi.llm.engine import Engine
from najdi.llm.output import parse_turns, strip_thinking
from najdi.notify import telegram
from najdi.text.arabic import phrase_hits, word_count
from najdi.text.dedup import is_near_dup, shingles
from najdi.text.safety import sensitive_hits
from najdi.voho import load_dialogues

log = logging.getLogger(__name__)


class State:
    def __init__(self, kept: list[dict], attempts: list[dict]):
        self.kept, self.attempts = kept, attempts
        self.done = {r["id"] for r in kept} | {a["task_id"] for a in attempts}
        self.openers: dict[str, Counter] = defaultdict(Counter)
        self.pools: dict[str, list[set[str]]] = defaultdict(list)
        for row in kept:
            self.add(row)

    def add(self, row: dict) -> None:
        for o in row_openers(row["messages"]):
            self.openers[row["subset"]][o] += 1
        self.pools[row["subset"]].append(shingles(dedup_text(row["subset"], row["messages"])))


class Generator:
    def __init__(
        self,
        cfg: Config,
        engine: Engine,
        classifier: DialectClassifier,
        prompts: PromptBuilder,
        rules: RuleFilter,
        judge: Judge,
    ):
        self.cfg = cfg
        self.engine = engine
        self.classifier = classifier
        self.prompts = prompts
        self.rules = rules
        self.judge = judge
        self.gen_extra = cfg.thinking_extra_tokens if cfg.thinking != "off" else 0
        self.judge_extra = cfg.thinking_extra_tokens if cfg.judge_thinking != "off" else 0

    def avoid(self, state: State, subset: str) -> list[str]:
        cap = self.rules.opener_cap(subset)
        return [o for o, c in state.openers[subset].most_common(15) if c >= max(2, cap // 2)]

    def row(self, t: dict, msgs: list[dict]) -> dict:
        seed = {k: v for k, v in t.items() if k != "anchors"}
        return {
            "id": t["id"],
            "subset": t["subset"],
            "topic": t["topic"],
            "seed": json.dumps(seed, ensure_ascii=False),
            "generator": self.cfg.engine.model,
            "messages": msgs,
        }

    def batch(self, tasks: list[dict], state: State) -> tuple[list[dict], list[dict]]:
        f = self.cfg.filters
        convs = [self.prompts.messages(t, self.avoid(state, t["subset"])) for t in tasks]
        params = [self.engine.json_sampling(self.prompts.schema(t), self.cfg.sampling, self.gen_extra) for t in tasks]
        outs = self.engine.chat(convs, params, thinking=self.cfg.thinking)

        recs = []
        for t, (text, finish) in zip(tasks, outs):
            rec = {"task_id": t["id"], "subset": t["subset"], "topic": t["topic"], "status": "dropped", "reason": None}
            answer = strip_thinking(text)
            parsed = parse_turns(answer, t["turns"])
            if parsed is None:
                rec.update(reason="format", finish=finish, raw=answer[:3000])
            else:
                msgs = parsed[0]
                reason, matched = self.rules.check(t, msgs)
                rec.update(messages=msgs, reason=reason)
                if matched:
                    rec["matched"] = matched
            recs.append(rec)

        def judged(m: dict) -> bool:
            return m["role"] == "assistant" and word_count(m["content"]) >= f.classifier_min_words

        alive = [r for r in recs if r["reason"] is None]
        replies = [m["content"] for r in alive for m in r["messages"] if judged(m)]
        labels = iter(self.classifier.labels(replies) if replies else [])
        for r in alive:
            r["dialect"] = [next(labels) for m in r["messages"] if judged(m)]
            if any(label != "GLF" for label in r["dialect"]):
                r["reason"] = "dialect_clf"

        batch_pools: dict[str, list[set[str]]] = defaultdict(list)
        for r in recs:
            if r["reason"] is None:
                sh = shingles(dedup_text(r["subset"], r["messages"]))
                pool = batch_pools[r["subset"]]
                if is_near_dup(sh, state.pools[r["subset"]], f.dedup_jaccard) or is_near_dup(sh, pool, f.dedup_jaccard):
                    r["reason"] = "dedup"
                else:
                    pool.append(sh)

        todo = [(t, r) for t, r in zip(tasks, recs) if r["reason"] is None]
        if todo:
            convs = [self.judge.messages(t, r["messages"]) for t, r in todo]
            params = [
                self.engine.json_sampling(self.judge.schema(t), self.cfg.judge_sampling, self.judge_extra)
                for t, _ in todo
            ]
            outs = self.engine.chat(convs, params, thinking=self.cfg.judge_thinking)
            for (t, r), (text, _) in zip(todo, outs):
                failed, verdict = self.judge.failures(t, strip_thinking(text))
                r["judge"] = verdict
                if failed is None:
                    r.update(reason="judge", judge_failed=["no verdict"])
                elif failed:
                    r.update(reason="judge", judge_failed=failed)

        rows = []
        for t, r in zip(tasks, recs):
            if r["reason"] is not None:
                continue
            cap = self.rules.opener_cap(t["subset"])
            full = [o for o in row_openers(r["messages"]) if state.openers[t["subset"]][o] >= cap]
            if full:
                r.update(reason="opener_variety", matched=full)
                continue
            r["status"] = "kept"
            row = self.row(t, r["messages"])
            rows.append(row)
            state.add(row)
        return rows, recs


def plan_batch(need: dict[str, int], attempts: list[dict], batch_tasks: int) -> dict[str, int]:
    want = {}
    for s, n in need.items():
        tried = [a for a in attempts if a["subset"] == s]
        rate = (sum(a["status"] == "kept" for a in tried) + 1) / (len(tried) + 2)
        want[s] = math.ceil(n / max(rate, 0.1)) if n else 0
    scale = min(1.0, batch_tasks / max(sum(want.values()), 1))
    return {s: math.ceil(w * scale) for s, w in want.items()}


def load_anchors(voho_id: str) -> dict[str, list[list[dict]]]:
    def clean(msgs: list[dict]) -> bool:
        text = " ".join(m["content"] for m in msgs)
        return not sensitive_hits(text) and not phrase_hits(text, INTIMATE)

    rows = load_dialogues(voho_id)
    return {
        s: [r["messages"] for r in rows if r["set"] == s and clean(r["messages"])] for s in ("enterprise", "everyday")
    }


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        root = Path(cfg.output_dir)
        out_dir = root / cfg.engine.model.split("/")[-1] / ("pilot" if cfg.pilot else "full")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.data_path = out_dir / "data.jsonl"
        self.attempts_path = out_dir / "attempts.jsonl"
        self.places = Places(root / "places.json", cfg.place_summary_words, cfg.places_per_prompt)
        builder = SeedBuilder(cfg.seed)
        self.seeds = {s: builder.build(s, cfg.max_attempts_factor * n) for s, n in cfg.targets.items()}
        log.info("seeds: %s", {s: len(v) for s, v in self.seeds.items()})
        anchors = load_anchors(cfg.voho_id)
        engine = Engine(cfg.engine, cfg.seed)
        self.prompts = PromptBuilder(cfg, anchors, self.places)
        self.generator = Generator(
            cfg,
            engine,
            DialectClassifier(cfg.filters.dialect_classifier),
            self.prompts,
            RuleFilter(cfg, self.places),
            Judge(cfg, self.places),
        )
        self.name = f"chat data {'pilot' if cfg.pilot else 'run'} ({cfg.engine.model.split('/')[-1]})"

    def push(self, state: State, message: str) -> bool:
        files = {}
        for subset in self.cfg.targets:
            rows = [r for r in state.kept if r["subset"] == subset]
            if rows:
                files[f"data/{subset}.jsonl"] = jsonl.write(self.out_dir / "hub" / f"{subset}.jsonl", rows)
        files["README.md"] = dataset_card(self.cfg, state.kept, state.attempts, self.places)
        try:
            repo_id = hub.push(self.cfg.repo_id, files, message)
        except Exception as exc:
            log.warning("hub push failed: %s", exc)
            return False
        log.info("pushed %d rows to %s", len(state.kept), repo_id)
        return True

    def goal(self, subsets: list[str]) -> dict[str, int]:
        targets = self.cfg.targets
        if not self.cfg.pilot:
            return {s: targets[s] for s in subsets}
        total = sum(targets[s] for s in subsets)
        goal = {s: round(self.cfg.pilot_rows * targets[s] / total) for s in subsets}
        goal[subsets[0]] += self.cfg.pilot_rows - sum(goal.values())
        return goal

    def run_part(self, part: str, subsets: list[str], state: State, progress: dict) -> None:
        cfg = self.cfg
        goal = self.goal(subsets)
        name = f"part {part} ({', '.join(subsets)})"
        log.info("%s | goal %s", name, goal)
        telegram(f"{self.name}: {name} started, goal {goal}.")
        while True:
            have = Counter(r["subset"] for r in state.kept)
            tried = Counter(a["subset"] for a in state.attempts)
            need = {s: max(0, goal[s] - have[s]) for s in subsets}
            if cfg.pilot:
                if any(tried[s] for s in subsets):
                    break
                plan = dict(goal)
            elif not any(need.values()):
                break
            else:
                plan = plan_batch(need, state.attempts, cfg.batch_tasks)
            tasks = []
            for s, k in plan.items():
                pending = [x for x in self.seeds[s] if x["id"] not in state.done][:k]
                tasks += [self.prompts.task(x) for x in pending]
            if not tasks:
                log.warning("%s: all seeds used", name)
                break
            rows, recs = self.generator.batch(tasks, state)
            jsonl.append(self.data_path, rows)
            jsonl.append(self.attempts_path, recs)
            state.kept += rows
            state.attempts += recs
            state.done |= {t["id"] for t in tasks}
            have = Counter(r["subset"] for r in state.kept)
            minutes = (time.time() - progress["t0"]) / 60
            dropped = Counter(r["reason"] for r in recs if r["status"] == "dropped")
            log.info(
                "[%.0f min] batch kept %d/%d | %s | dropped: %s",
                minutes,
                len(rows),
                len(tasks),
                ", ".join(f"{s} {have[s]}/{goal[s]}" for s in subsets),
                ", ".join(f"{k} {v}" for k, v in dropped.most_common()),
            )
            bucket = len(state.kept) // cfg.push_every
            if (
                not cfg.pilot
                and bucket > progress["pushed_bucket"]
                and self.push(state, f"checkpoint: {len(state.kept)} rows")
            ):
                progress["pushed_bucket"], progress["pushed_rows"] = bucket, len(state.kept)
            if len(state.kept) // cfg.telegram_every > progress["told_bucket"]:
                progress["told_bucket"] = len(state.kept) // cfg.telegram_every
                telegram(f"{self.name}: {len(state.kept)} rows kept so far, {minutes:.0f} min.")
        removed = Counter(a["reason"] for a in state.attempts if a["subset"] in subsets and a["status"] == "dropped")
        telegram(f"{self.name}: {name} finished. Removed: " + ", ".join(f"{k} {v}" for k, v in removed.most_common()))

    def run(self) -> None:
        cfg = self.cfg
        if cfg.pilot:
            for p in (self.data_path, self.attempts_path):
                p.unlink(missing_ok=True)
        for p in (self.data_path, self.attempts_path):
            jsonl.repair(p)
        state = State(jsonl.read(self.data_path), jsonl.read(self.attempts_path))
        progress = {
            "t0": time.time(),
            "pushed_bucket": len(state.kept) // cfg.push_every,
            "pushed_rows": None,
            "told_bucket": len(state.kept) // cfg.telegram_every,
        }
        try:
            for part, subsets in cfg.parts.items():
                self.run_part(part, subsets, state, progress)
        except Exception as exc:
            telegram(f"{self.name} crashed with {len(state.kept)} rows kept: {type(exc).__name__}: {exc}")
            raise
        for subset in cfg.targets:
            recs = [a for a in state.attempts if a["subset"] == subset]
            for name, reached, k in filter_table(recs):
                if k:
                    log.info("%s %-16s removed %4d of %d", subset, name, k, reached)
        for line in coverage(state.kept):
            log.info(line)
        if not cfg.pilot and progress["pushed_rows"] != len(state.kept):
            self.push(state, f"final: {len(state.kept)} rows")
        telegram(f"{self.name} finished: {len(state.kept)} rows kept from {len(state.attempts)} generated.")


def run(cfg: Config) -> None:
    Runner(cfg).run()
