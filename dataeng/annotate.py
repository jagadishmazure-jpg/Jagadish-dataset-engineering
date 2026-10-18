"""Annotation workforce pipeline: task assignment, label collection, adjudication,
and labeler-skew detection.

Every label cites the (task_id, labeler_id) pair that produced it, so any
adjudicated outcome is traceable back to its human sources.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


def chi2_sf(x: float, df: int) -> float:
    """P(X >= x) for a chi-square variable with ``df`` degrees of freedom.

    Regularized upper incomplete gamma Q(df/2, x/2): series for small x,
    Lentz continued fraction otherwise (Numerical Recipes 6.2). Stdlib only.
    """
    if df < 1:
        raise ValueError("df must be >= 1")
    if x <= 0:
        return 1.0
    a, z = df / 2.0, x / 2.0
    gln = math.lgamma(a)
    if z < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(500):
            ap += 1
            term *= z / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - gln))
    tiny = 1e-300
    b = z + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(-z + a * math.log(z) - gln) * h)


@dataclass(frozen=True)
class Label:
    """A single human judgement. The (task_id, labeler_id) citation is the
    audit trail: nothing downstream may use a label without it."""

    task_id: str
    labeler_id: str
    label: str


@dataclass
class Task:
    task_id: str
    text: str
    options: list


@dataclass
class AdjudicationResult:
    task_id: str
    label: str | None  # None when tied / unlabeled
    votes: dict  # label -> count
    status: str  # "resolved" | "tie" | "unlabeled"
    labeler_ids: list  # every labeler whose vote is cited


class AnnotationPipeline:
    """Round-robin task assignment with configurable overlap, majority-vote
    adjudication, and chi-square-style labeler skew detection."""

    def __init__(self, labelers: list, overlap: int = 2,
                 skew_threshold: float = 10.0, skew_min_labels: int = 6):
        if len(labelers) < 2:
            raise ValueError("need at least 2 labelers")
        if not 1 <= overlap <= len(labelers):
            raise ValueError("overlap must be between 1 and number of labelers")
        self.labelers = list(labelers)
        self.overlap = overlap
        self.skew_threshold = skew_threshold
        self.skew_min_labels = skew_min_labels
        self.tasks: dict[str, Task] = {}
        self.assignments: dict[str, list[str]] = {}  # task_id -> [labeler_id]
        self.labels: list[Label] = []

    # -- task management -------------------------------------------------
    def add_task(self, task_id: str, text: str, options: list) -> None:
        if task_id in self.tasks:
            raise ValueError(f"duplicate task_id: {task_id}")
        if len(options) < 2:
            raise ValueError("task needs at least 2 label options")
        self.tasks[task_id] = Task(task_id, text, list(options))

    def assign(self) -> dict[str, list[str]]:
        """Deterministic rotating assignment: task i goes to labelers
        (i, i+1, ..., i+overlap-1) mod n. Every task gets exactly `overlap`
        labelers; every labeler gets a balanced share."""
        task_ids = sorted(self.tasks)
        n = len(self.labelers)
        self.assignments = {}
        for i, tid in enumerate(task_ids):
            self.assignments[tid] = [self.labelers[(i + j) % n] for j in range(self.overlap)]
        return {tid: list(lids) for tid, lids in self.assignments.items()}

    # -- labeling --------------------------------------------------------
    def submit_label(self, task_id: str, labeler_id: str, label: str) -> Label:
        if task_id not in self.tasks:
            raise ValueError(f"unknown task_id: {task_id}")
        task = self.tasks[task_id]
        if label not in task.options:
            raise ValueError(
                f"label {label!r} not in options {task.options} for task {task_id}"
            )
        assigned = self.assignments.get(task_id, [])
        if assigned and labeler_id not in assigned:
            raise ValueError(
                f"labeler {labeler_id} not assigned to task {task_id} "
                f"(assigned: {assigned})"
            )
        if labeler_id not in self.labelers:
            raise ValueError(f"unknown labeler_id: {labeler_id}")
        lab = Label(task_id=task_id, labeler_id=labeler_id, label=label)
        self.labels.append(lab)
        return lab

    # -- adjudication ------------------------------------------------------
    def adjudicate(self) -> dict[str, AdjudicationResult]:
        by_task: dict[str, list[Label]] = {}
        for lab in self.labels:
            by_task.setdefault(lab.task_id, []).append(lab)
        results = {}
        for tid in sorted(self.tasks):
            labs = by_task.get(tid, [])
            if not labs:
                results[tid] = AdjudicationResult(tid, None, {}, "unlabeled", [])
                continue
            votes = Counter(l.label for l in labs)
            top = votes.most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                results[tid] = AdjudicationResult(
                    tid, None, dict(votes), "tie",
                    sorted({l.labeler_id for l in labs}),
                )
            else:
                results[tid] = AdjudicationResult(
                    tid, top[0][0], dict(votes), "resolved",
                    sorted({l.labeler_id for l in labs}),
                )
        return results

    # -- skew detection ----------------------------------------------------
    def detect_skew(self) -> list[dict]:
        """Flag labelers whose label distribution diverges from their peers.

        Hand-rolled chi-square: for each labeler, compare their observed label
        counts against the pooled distribution of *all other* labelers (with
        Laplace smoothing). Flags when chi2 > skew_threshold.
        """
        by_labeler: dict[str, Counter] = {lid: Counter() for lid in self.labelers}
        for lab in self.labels:
            by_labeler[lab.labeler_id][lab.label] += 1

        vocab = sorted({lab.label for lab in self.labels})
        if len(vocab) < 2:
            return [
                {"labeler_id": lid, "n_labels": sum(by_labeler[lid].values()),
                 "chi2": 0.0, "threshold": self.skew_threshold,
                 "flagged": False, "reason": "insufficient label variety"}
                for lid in self.labelers
            ]

        results = []
        for lid in self.labelers:
            obs = by_labeler[lid]
            n = sum(obs.values())
            if n < self.skew_min_labels:
                results.append({
                    "labeler_id": lid, "n_labels": n, "chi2": 0.0,
                    "threshold": self.skew_threshold, "flagged": False,
                    "reason": f"fewer than {self.skew_min_labels} labels",
                })
                continue
            # pooled peer distribution with Laplace smoothing
            peer = Counter()
            for other, cnt in by_labeler.items():
                if other != lid:
                    peer.update(cnt)
            peer_total = sum(peer.values())
            alpha = 1.0
            exp_probs = {
                v: (peer.get(v, 0) + alpha) / (peer_total + alpha * len(vocab))
                for v in vocab
            }
            chi2 = sum(
                (obs.get(v, 0) - n * exp_probs[v]) ** 2 / (n * exp_probs[v] + 1e-9)
                for v in vocab
            )
            results.append({
                "labeler_id": lid,
                "n_labels": n,
                "chi2": round(chi2, 3),
                "threshold": self.skew_threshold,
                "flagged": chi2 > self.skew_threshold,
                # df and p-value make the fixed threshold interpretable: the
                # same chi2 means less with more label options.
                "df": len(vocab) - 1,
                "p_value": round(chi2_sf(chi2, len(vocab) - 1), 6),
                "distribution": {v: obs.get(v, 0) for v in vocab},
            })
        return results
