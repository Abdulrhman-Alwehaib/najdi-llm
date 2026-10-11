from __future__ import annotations

import hashlib
import re
import zlib
from collections import defaultdict

from najdi.text.arabic import LETTERS, normalize

_NOT_LETTER = re.compile(f"[^{LETTERS}0-9]+")


def shingles(text: str, n: int = 4) -> set[str]:
    s = " ".join(_NOT_LETTER.sub(" ", normalize(text)).split())
    return {s[i : i + n] for i in range(max(1, len(s) - n + 1))}


def is_near_dup(sh: set[str], pool: list[set[str]], threshold: float) -> bool:
    for other in pool:
        inter = len(sh & other)
        if inter and inter >= threshold * (len(sh) + len(other) - inter):
            return True
    return False


class MinHashIndex:
    def __init__(self, threshold: float, num_perm: int = 128, bands: int = 32, seed: int = 1234, ngram: int = 5):
        import numpy as np

        self.np = np
        self.threshold = threshold
        self.ngram = ngram
        self.bands = bands
        self.per_band = num_perm // bands
        self.prime = np.uint64(4294967311)
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2**32, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 2**32, size=num_perm, dtype=np.uint64)
        self.digests: set[str] = set()
        self.buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
        self.sets: list[set[str]] = []

    @staticmethod
    def key(text: str) -> str:
        return " ".join(re.sub(r"[^\w]+", " ", normalize(text).lower()).split())

    def _signature(self, grams: set[str]):
        np = self.np
        x = np.fromiter((zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint64, count=len(grams))
        return ((np.outer(x, self.a) + self.b) % self.prime).min(axis=0)

    def add(self, text: str) -> str | None:
        key = self.key(text)
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        if digest in self.digests:
            return "exact"
        grams = {key[i : i + self.ngram] for i in range(max(1, len(key) - self.ngram + 1))}
        sig = self._signature(grams)
        bands = [(b, sig[b * self.per_band : (b + 1) * self.per_band].tobytes()) for b in range(self.bands)]
        for i in {i for band in bands for i in self.buckets[band]}:
            inter = len(grams & self.sets[i])
            if inter >= self.threshold * (len(grams) + len(self.sets[i]) - inter):
                return "near"
        self.digests.add(digest)
        for band in bands:
            self.buckets[band].append(len(self.sets))
        self.sets.append(grams)
        return None
