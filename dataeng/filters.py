"""Heuristic quality filters for raw conversation logs.

Each check is independent and named; a filter run reports exactly which
checks failed so rejections are auditable. The language check is an explicit
stub (wordlist ratio) — production would swap in a real identifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# A small common-word list for the language stub. Presence of ordinary words
# is a cheap, deterministic English-likeness signal.
COMMON_WORDS = frozenset(
    """the be to of and a in that have i it for not on with he as you do at
    this but his by from they we say her she or an will my one all would there
    their what so up out if about who get which go me when make can like time
    just him know take people into year your good some could them see other
    than then now look only come its over think also back after use two how
    our work first well way even new want because any these give day most us
    hello thanks please help need want order account refund return shipping
    yes no not very much more""".split()
)

PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
}


def _word_trigrams(words: list[str]) -> list[tuple]:
    return [tuple(words[i:i + 3]) for i in range(len(words) - 2)]


@dataclass
class FilterResult:
    passed: bool
    failed: list  # names of checks that failed
    details: dict = field(default_factory=dict)


class ConversationFilter:
    """Configurable heuristic gate for raw conversation text."""

    def __init__(self, min_chars: int = 20, max_chars: int = 4000,
                 min_english_ratio: float = 0.25,
                 min_trigram_uniqueness: float = 0.35,
                 max_char_run: int = 15):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_english_ratio = min_english_ratio
        self.min_trigram_uniqueness = min_trigram_uniqueness
        self.max_char_run = max_char_run

    # -- individual checks ------------------------------------------------
    def check_length(self, text: str) -> tuple[bool, str]:
        n = len(text)
        if n < self.min_chars:
            return False, f"too short ({n} < {self.min_chars} chars)"
        if n > self.max_chars:
            return False, f"too long ({n} > {self.max_chars} chars)"
        return True, f"{n} chars within bounds"

    def check_language(self, text: str) -> tuple[bool, str]:
        """Stub: fraction of alphabetic tokens in COMMON_WORDS."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if not words:
            return False, "no alphabetic tokens"
        ratio = sum(1 for w in words if w in COMMON_WORDS) / len(words)
        if ratio < self.min_english_ratio:
            return False, f"english-word ratio {ratio:.2f} < {self.min_english_ratio}"
        return True, f"english-word ratio {ratio:.2f}"

    def check_pii(self, text: str) -> tuple[bool, str]:
        hits = [name for name, rx in PII_PATTERNS.items() if rx.search(text)]
        if hits:
            return False, f"PII patterns detected: {', '.join(hits)}"
        return True, "no PII patterns"

    def check_repetition(self, text: str) -> tuple[bool, str]:
        # character-run repetition, e.g. "aaaaaaa..." or "!!!!!!!"
        if re.search(r"(.)\1{" + str(self.max_char_run) + r",}", text):
            return False, f"character repeated >{self.max_char_run}x consecutively"
        words = text.lower().split()
        trigrams = _word_trigrams(words)
        if len(trigrams) >= 5:
            uniq = len(set(trigrams)) / len(trigrams)
            if uniq < self.min_trigram_uniqueness:
                return False, (
                    f"trigram uniqueness {uniq:.2f} < {self.min_trigram_uniqueness}"
                )
            return True, f"trigram uniqueness {uniq:.2f}"
        return True, "too few trigrams to judge repetition"

    # -- combined ----------------------------------------------------------
    CHECKS = ("length", "language", "pii", "repetition")

    def check(self, text: str) -> FilterResult:
        failed, details = [], {}
        for name in self.CHECKS:
            ok, detail = getattr(self, f"check_{name}")(text)
            details[name] = detail
            if not ok:
                failed.append(name)
        return FilterResult(passed=not failed, failed=failed, details=details)

    def filter_batch(self, items: list[tuple[str, str]]) -> tuple[list, list]:
        """items: [(item_id, text)]. Returns (kept, rejected) where rejected
        entries are (item_id, FilterResult)."""
        kept, rejected = [], []
        for item_id, text in items:
            res = self.check(text)
            (kept if res.passed else rejected).append(
                item_id if res.passed else (item_id, res)
            )
        return kept, rejected
