"""Synthetic data generation with QC gates.

Templates with {placeholders} are filled from seed pools. Every candidate
passes through gates before acceptance; rejections are logged with reasons
so the generator's failure modes are visible, not silent:

  - schema:   all placeholders filled, no leftover braces, fields non-empty
  - length:   meets minimum character length
  - dedup:    not an exact/near duplicate of existing data or the batch so far
  - diversity: no single template may dominate the batch (max share cap)
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

from .dedup import Deduplicator

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass
class GeneratedItem:
    item_id: str
    text: str
    template: str
    fields: dict


@dataclass
class Rejection:
    candidate_text: str
    template: str
    reasons: list


class SyntheticGenerator:
    def __init__(self, templates: list[str], pools: dict[str, list[str]],
                 seed: int = 42, min_chars: int = 30,
                 max_template_share: float = 0.5,
                 jaccard_threshold: float = 0.85):
        if not templates:
            raise ValueError("need at least one template")
        self.templates = list(templates)
        self.pools = {k: list(v) for k, v in pools.items()}
        self.seed = seed
        self.min_chars = min_chars
        self.max_template_share = max_template_share
        self.jaccard_threshold = jaccard_threshold

    def _render(self, template: str, rng: random.Random) -> tuple[str | None, dict, str | None]:
        """Returns (text, fields, error). error is a schema-failure reason."""
        fields = {}
        for name in sorted(set(_PLACEHOLDER.findall(template))):
            pool = self.pools.get(name)
            if not pool:
                return None, {}, f"schema: no pool for placeholder {{{name}}}"
            fields[name] = rng.choice(pool)
        try:
            text = template.format(**fields)
        except (KeyError, IndexError, ValueError):
            return None, {}, "schema: template formatting failed"
        if _PLACEHOLDER.search(text):
            return None, {}, "schema: unfilled placeholder remains"
        if not all(str(v).strip() for v in fields.values()):
            return None, {}, "schema: empty field value"
        return text, fields, None

    def generate(self, n: int,
                 existing_texts: tuple[str, ...] = ()) -> tuple[list, list]:
        """Generate up to n accepted items. Returns (items, rejections)."""
        rng = random.Random(self.seed)
        dedup = Deduplicator(jaccard_threshold=self.jaccard_threshold,
                             seed=self.seed)
        for i, t in enumerate(existing_texts):
            dedup.add(f"existing-{i}", t)

        items: list[GeneratedItem] = []
        rejections: list[Rejection] = []
        template_counts: dict[str, int] = {}
        attempts = 0
        max_attempts = max(n * 20, 50)
        # Diversity is enforced against the *target* batch size: no template
        # may contribute more than ceil(share * n) items. (Checking against
        # the running total would wrongly reject the very first item, whose
        # single-template share is trivially 100%.)
        template_cap = math.ceil(self.max_template_share * n)

        while len(items) < n and attempts < max_attempts:
            attempts += 1
            template = self.templates[rng.randrange(len(self.templates))]
            text, fields, error = self._render(template, rng)
            reasons = []
            if error:
                reasons.append(error)
            else:
                if len(text) < self.min_chars:
                    reasons.append(
                        f"length: {len(text)} chars < minimum {self.min_chars}"
                    )
                if dedup.exact_match(text) is not None:
                    reasons.append("dedup: exact duplicate of existing data")
                elif dedup.query(text):
                    reasons.append("dedup: near-duplicate of existing data")
                # diversity: would this template exceed its share of the batch?
                prospective = template_counts.get(template, 0) + 1
                if prospective > template_cap:
                    reasons.append(
                        f"diversity: template would exceed "
                        f"{self.max_template_share} share "
                        f"(>{template_cap} of {n})"
                    )
            if reasons:
                rejections.append(Rejection(text or "", template, reasons))
                continue
            item_id = f"syn-{len(items):05d}"
            items.append(GeneratedItem(item_id, text, template, fields))
            template_counts[template] = template_counts.get(template, 0) + 1
            dedup.add(item_id, text)

        return items, rejections
