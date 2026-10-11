from __future__ import annotations

import hashlib
import logging
import random
from collections import defaultdict

from najdi.mix.config import SOURCES, Config, counts_line

log = logging.getLogger(__name__)

TEST_GROUPS = {
    "science_logic": ["science", "math_logic"],
    "voho": ["voho_everyday", "voho_enterprise"],
    "fusha": ["fusha"],
}


class Splitter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def rank(self, r: dict) -> str:
        return hashlib.md5(f"{self.cfg.seed}:{r['id']}".encode()).hexdigest()

    @staticmethod
    def allocate(n: int, weights: dict) -> dict:
        total = sum(weights.values())
        if not total:
            return {k: 0 for k in weights}
        raw = {k: n * w / total for k, w in weights.items()}
        out = {k: int(v) for k, v in raw.items()}
        for k in sorted(raw, key=lambda k: (out[k] - raw[k], k))[: n - sum(out.values())]:
            out[k] += 1
        return out

    def _by_stratum(self, rows: list[dict]) -> list[list[dict]]:
        groups = defaultdict(list)
        for r in sorted(rows, key=self.rank):
            groups[r["_strata"]].append(r)
        return [groups[k] for k in sorted(groups, key=lambda k: self.rank(groups[k][0]))]

    def spread_pick(self, rows: list[dict], n: int) -> list[dict]:
        queues, picked, depth = self._by_stratum(rows), [], 0
        while len(picked) < n:
            layer = [q[depth] for q in queues if depth < len(q)]
            if not layer:
                break
            picked += layer[: n - len(picked)]
            depth += 1
        return picked

    def proportional_pick(self, rows: list[dict], n: int) -> list[dict]:
        queues = self._by_stratum(rows)
        quota = self.allocate(min(n, len(rows)), {i: len(q) for i, q in enumerate(queues)})
        return [r for i, q in enumerate(queues) for r in q[: quota[i]]]

    def science_pick(self, rows: list[dict], n: int) -> list[dict]:
        keep = set(self.cfg.always_keep_topics)
        must = [r for r in sorted(rows, key=self.rank) if r["_strata"] in keep][:n]
        rest = [r for r in rows if r["_strata"] not in keep]
        return must + self.proportional_pick(rest, n - len(must))

    def targets(self, pool: dict[str, list[dict]]) -> dict[str, int]:
        cfg = self.cfg
        available = {s: len(pool[s]) for s in ("science", "math_logic")}
        target = self.allocate(min(cfg.science_n, sum(available.values())), available)
        target["helpful"], target["identity"] = len(pool["helpful"]), len(pool["identity"])
        for s, wanted in {
            "voho_everyday": cfg.voho_everyday_n,
            "voho_enterprise": cfg.voho_enterprise_n,
            "fusha": cfg.fusha_n,
        }.items():
            target[s] = min(wanted, len(pool[s]))
            if target[s] < wanted:
                log.warning("only %d %s rows are available (wanted %d)", target[s], s, wanted)
        return target

    def __call__(self, rows: list[dict]) -> tuple[dict[str, list[dict]], dict[str, int]]:
        cfg = self.cfg
        pool = defaultdict(list)
        for r in rows:
            pool[r["source"]].append(r)
        target = self.targets(pool)

        splits = {"train": [], "validation": [], "test": []}
        test_n = {s: 0 for s in SOURCES}
        for group, sources in TEST_GROUPS.items():
            test_n.update(self.allocate(cfg.test_n.get(group, 0), {s: target[s] for s in sources}))
        for s in SOURCES:
            picked = self.spread_pick(pool[s], test_n[s])
            splits["test"] += picked
            taken = {r["id"] for r in picked}
            pool[s] = [r for r in pool[s] if r["id"] not in taken]

        for s in SOURCES:
            need = max(0, target[s] - test_n[s])
            pick = self.science_pick if s in ("science", "math_logic") else self.spread_pick
            mix = pick(pool[s], need)
            val = self.spread_pick(mix, round(cfg.val_frac * len(mix)))
            val_ids = {r["id"] for r in val}
            splits["validation"] += val
            train = [r for r in mix if r["id"] not in val_ids]
            if s == "helpful":
                train = [
                    r if k == 0 else {**r, "id": f"{r['id']}-copy{k + 1}"}
                    for k in range(cfg.helpful_repeat)
                    for r in train
                ]
            splits["train"] += train
        random.Random(cfg.seed).shuffle(splits["train"])

        ids = {name: {r["id"] for r in part} for name, part in splits.items()}
        if any(len(ids[name]) != len(part) for name, part in splits.items()):
            raise RuntimeError("ids must be unique within a split")
        base = {name: {i.split("-copy")[0] for i in ids[name]} for name in ids}
        if base["test"] & base["train"] or base["test"] & base["validation"] or base["train"] & base["validation"]:
            raise RuntimeError("a row is in more than one split")
        unique = {
            s: len({r["id"].split("-copy")[0] for part in splits.values() for r in part if r["source"] == s})
            for s in SOURCES
        }
        for name, part in splits.items():
            log.info("%-10s %s", name, counts_line(part))
        return splits, unique
