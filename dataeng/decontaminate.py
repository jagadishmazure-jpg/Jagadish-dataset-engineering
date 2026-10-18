"""Eval-set decontamination: strip train items that match eval items.

Eval leakage (train items that are copies or near-copies of eval items)
inflates benchmark scores and hides real generalization gaps. This module
removes them and reports exactly what was removed, against which eval item,
with a match score — so the decontamination itself is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .dedup import Deduplicator


@dataclass
class DecontaminationReport:
    removed: list = field(default_factory=list)
    # each: {"train_id", "eval_id", "score", "kind": "exact"|"near"}
    clean_train_ids: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def decontaminate(train: list[tuple[str, str]],
                  eval_items: list[tuple[str, str]],
                  jaccard_threshold: float = 0.5,
                  seed: int = 12345) -> DecontaminationReport:
    """Remove train items matching any eval item (exact or near-duplicate).

    train / eval_items: [(item_id, text)]. Returns a report with the removed
    items (each citing the eval item it matched and a score) and the surviving
    train ids, in original order.
    """
    dedup = Deduplicator(jaccard_threshold=jaccard_threshold, seed=seed)
    for eid, etext in eval_items:
        dedup.add(eid, etext)

    removed = []
    removed_ids = set()
    for tid, ttext in train:
        exact = dedup.exact_match(ttext)
        if exact is not None:
            removed.append({"train_id": tid, "eval_id": exact,
                            "score": 1.0, "kind": "exact"})
            removed_ids.add(tid)
            continue
        hits = dedup.query(ttext)
        if hits:
            eid, score = hits[0]
            removed.append({"train_id": tid, "eval_id": eid,
                            "score": score, "kind": "near"})
            removed_ids.add(tid)

    clean = [tid for tid, _ in train if tid not in removed_ids]
    kinds = {}
    for r in removed:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    summary = {
        "n_train": len(train),
        "n_eval": len(eval_items),
        "n_removed": len(removed),
        "n_clean": len(clean),
        "by_kind": kinds,
    }
    return DecontaminationReport(removed=removed, clean_train_ids=clean,
                                 summary=summary)
