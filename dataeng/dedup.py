"""Near-duplicate detection: hand-rolled MinHash + LSH banding, stdlib only.

- Exact duplicates take a fast path (normalized-text hash), no signatures needed.
- Near duplicates: k-shingle sets -> MinHash signatures (md5-based, seeded, so
  deterministic across runs) -> LSH banding proposes candidates -> exact Jaccard
  similarity confirms pairs above threshold -> union-find merges clusters.
"""
from __future__ import annotations

import hashlib
import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _md5_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


class Deduplicator:
    def __init__(self, num_perm: int = 64, bands: int = 16, shingle_k: int = 5,
                 jaccard_threshold: float = 0.5, seed: int = 12345):
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.shingle_k = shingle_k
        self.jaccard_threshold = jaccard_threshold
        self.seed = seed
        self._texts: dict[str, str] = {}          # id -> normalized text
        self._shingles: dict[str, set] = {}        # id -> shingle set
        self._signatures: dict[str, list] = {}     # id -> minhash signature
        self._exact: dict[str, str] = {}           # md5(normalized) -> first id

    # -- primitives --------------------------------------------------------
    def _shingle_set(self, text: str) -> set:
        tokens = normalize(text).split()
        k = self.shingle_k
        if len(tokens) < k:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}

    def _signature(self, shingles: set) -> list:
        sig = []
        for i in range(self.num_perm):
            best = None
            prefix = f"{self.seed}:{i}:"
            for sh in shingles:
                h = _md5_int(prefix + sh)
                if best is None or h < best:
                    best = h
            sig.append(best if best is not None else 0)
        return sig

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # -- public API ----------------------------------------------------------
    def add(self, item_id: str, text: str) -> None:
        if item_id in self._texts:
            raise ValueError(f"duplicate item_id: {item_id}")
        norm = normalize(text)
        self._texts[item_id] = norm
        shingles = self._shingle_set(text)
        self._shingles[item_id] = shingles
        self._signatures[item_id] = self._signature(shingles)
        key = hashlib.md5(norm.encode("utf-8")).hexdigest()
        self._exact.setdefault(key, item_id)

    def exact_match(self, text: str) -> str | None:
        """Return the id of an exactly-duplicate item, else None."""
        key = hashlib.md5(normalize(text).encode("utf-8")).hexdigest()
        return self._exact.get(key)

    def _candidates(self) -> set[tuple[str, str]]:
        buckets: dict[tuple, list] = {}
        for item_id, sig in self._signatures.items():
            for b in range(self.bands):
                band = tuple(sig[b * self.rows:(b + 1) * self.rows])
                buckets.setdefault((b, band), []).append(item_id)
        pairs = set()
        for ids in buckets.values():
            ids = sorted(set(ids))
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.add((ids[i], ids[j]))
        return pairs

    def find_duplicates(self) -> list[list[str]]:
        """Return clusters (sorted id lists) of exact + near duplicates."""
        parent = {i: i for i in self._texts}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # exact clusters
        by_hash: dict[str, list] = {}
        for item_id, norm in self._texts.items():
            key = hashlib.md5(norm.encode("utf-8")).hexdigest()
            by_hash.setdefault(key, []).append(item_id)
        for ids in by_hash.values():
            for other in ids[1:]:
                union(ids[0], other)

        # near-duplicate candidates confirmed by Jaccard
        for a, b in self._candidates():
            sim = self._jaccard(self._shingles[a], self._shingles[b])
            if sim >= self.jaccard_threshold:
                union(a, b)

        clusters: dict[str, list] = {}
        for item_id in self._texts:
            clusters.setdefault(find(item_id), []).append(item_id)
        return [sorted(ids) for ids in clusters.values() if len(ids) > 1]

    def query(self, text: str) -> list[tuple[str, float]]:
        """Return [(item_id, jaccard)] for stored items at/above threshold,
        sorted by similarity descending. Exact matches score 1.0."""
        norm = normalize(text)
        if not norm:
            return []
        qsh = self._shingle_set(text)
        hits = []
        for item_id, shingles in self._shingles.items():
            sim = self._jaccard(qsh, shingles)
            if sim >= self.jaccard_threshold:
                hits.append((item_id, round(sim, 3)))
        hits.sort(key=lambda h: (-h[1], h[0]))
        return hits
