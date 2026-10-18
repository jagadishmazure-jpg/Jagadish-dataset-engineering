"""Tests for dataeng: annotation, filters, dedup, decontamination, synthesis, cards.

All deterministic (seeded); no network; stdlib only.
"""
import pytest

from dataeng import (
    AnnotationPipeline,
    ConversationFilter,
    Deduplicator,
    decontaminate,
    SyntheticGenerator,
    build_dataset_card,
)


# ---------------------------------------------------------------- annotate
OPTIONS = ["positive", "neutral", "negative"]


def _skewed_pipeline():
    """12 tasks, 3 labelers, full overlap (everyone labels everything).
    ann-a and ann-b vote the rotating true label (balanced); ann-c always
    votes 'positive' (seeded skew)."""
    p = AnnotationPipeline(["ann-a", "ann-b", "ann-c"], overlap=3)
    for i in range(12):
        p.add_task(f"t-{i:02d}", f"rate sentence {i}", OPTIONS)
    p.assign()
    cycle = ["positive", "neutral", "negative"]
    for i in range(12):
        tid = f"t-{i:02d}"
        p.submit_label(tid, "ann-a", cycle[i % 3])
        p.submit_label(tid, "ann-b", cycle[i % 3])
        p.submit_label(tid, "ann-c", "positive")
    return p


def test_assignment_covers_every_task_with_overlap():
    p = AnnotationPipeline(["a", "b", "c"], overlap=2)
    for i in range(7):
        p.add_task(f"t{i}", "text", OPTIONS)
    assign = p.assign()
    assert all(len(lids) == 2 for lids in assign.values())
    # balanced load: 7 tasks * 2 / 3 labelers -> shares of 4 or 5
    counts = {}
    for lids in assign.values():
        for lid in lids:
            counts[lid] = counts.get(lid, 0) + 1
    assert sorted(counts.values()) == [4, 5, 5]


def test_assignment_is_deterministic():
    def build():
        p = AnnotationPipeline(["a", "b", "c", "d"], overlap=3)
        for i in range(10):
            p.add_task(f"t{i}", "text", OPTIONS)
        return p.assign()
    assert build() == build()


def test_submit_label_rejects_invalid_option():
    p = AnnotationPipeline(["a", "b"], overlap=2)
    p.add_task("t0", "text", OPTIONS)
    p.assign()
    with pytest.raises(ValueError):
        p.submit_label("t0", "a", "ecstatic")


def test_submit_label_rejects_unassigned_labeler():
    p = AnnotationPipeline(["a", "b", "c"], overlap=1)
    p.add_task("t0", "text", OPTIONS)
    p.assign()
    assigned = p.assignments["t0"]
    outsider = next(l for l in ["a", "b", "c"] if l not in assigned)
    with pytest.raises(ValueError):
        p.submit_label("t0", outsider, "positive")


def test_adjudicate_majority_vote():
    p = AnnotationPipeline(["a", "b", "c"], overlap=3)
    p.add_task("t0", "text", OPTIONS)
    p.assign()
    p.submit_label("t0", "a", "positive")
    p.submit_label("t0", "b", "positive")
    p.submit_label("t0", "c", "neutral")
    res = p.adjudicate()["t0"]
    assert res.status == "resolved"
    assert res.label == "positive"
    assert res.votes == {"positive": 2, "neutral": 1}
    assert res.labeler_ids == ["a", "b", "c"]


def test_adjudicate_tie_needs_expert():
    p = AnnotationPipeline(["a", "b"], overlap=2)
    p.add_task("t0", "text", OPTIONS)
    p.assign()
    p.submit_label("t0", "a", "positive")
    p.submit_label("t0", "b", "negative")
    res = p.adjudicate()["t0"]
    assert res.status == "tie"
    assert res.label is None


def test_detect_skew_flags_seeded_skewed_labeler():
    p = _skewed_pipeline()
    flags = {r["labeler_id"]: r["flagged"] for r in p.detect_skew()}
    assert flags["ann-c"] is True
    assert flags["ann-a"] is False
    assert flags["ann-b"] is False


def test_labels_cite_task_and_labeler():
    p = AnnotationPipeline(["a", "b"], overlap=2)
    p.add_task("t0", "some text", OPTIONS)
    p.assign()
    lab = p.submit_label("t0", "a", "neutral")
    assert lab.task_id == "t0" and lab.labeler_id == "a"
    res = p.adjudicate()["t0"]
    assert "a" in res.labeler_ids  # citation flows into adjudication


# ---------------------------------------------------------------- filters
def test_filter_length_bounds():
    f = ConversationFilter(min_chars=20, max_chars=100)
    assert "length" in f.check("too short").failed
    assert "length" in f.check("x" * 101).failed
    assert f.check("this is a perfectly fine length message").passed


def test_filter_pii_email():
    f = ConversationFilter()
    res = f.check("please email me at john.doe@example.com about my order")
    assert not res.passed and "pii" in res.failed
    assert "email" in res.details["pii"]


def test_filter_pii_phone_and_ssn():
    f = ConversationFilter()
    assert "pii" in f.check("call me at 415-555-0132 tomorrow").failed
    assert "pii" in f.check("my ssn is 123-45-6789 ok").failed
    assert "pii" in f.check("card 4111 1111 1111 1111 please").failed


def test_filter_pii_clean_text_passes():
    f = ConversationFilter()
    res = f.check("Hello, I need help with my refund for order five six seven")
    assert "pii" not in res.failed


def test_filter_language_stub():
    f = ConversationFilter()
    assert f.check("Hello, I need help with my order please").passed
    res = f.check("asdf qwer zxcv blorp fnord quux")
    assert "language" in res.failed


def test_filter_repetition_trigrams():
    f = ConversationFilter()
    spam = "buy now buy now buy now buy now buy now buy now buy now"
    assert "repetition" in f.check(spam).failed
    ok_text = ("Hello I need help with my order please and I want to "
               "know about the refund policy for damaged items")
    assert "repetition" not in f.check(ok_text).failed


def test_filter_repetition_char_run():
    f = ConversationFilter()
    assert "repetition" in f.check("this is great" + "!" * 20 + " thanks").failed


def test_filter_batch_splits_kept_and_rejected():
    f = ConversationFilter()
    items = [
        ("good-1", "Hello, I need help with my order please and thank you"),
        ("bad-pii", "my email is thief@example.com send it there now ok"),
        ("bad-short", "hi"),
    ]
    kept, rejected = f.filter_batch(items)
    assert kept == ["good-1"]
    assert [r[0] for r in rejected] == ["bad-pii", "bad-short"]
    by_id = {r[0]: r[1] for r in rejected}
    assert "pii" in by_id["bad-pii"].failed
    assert "length" in by_id["bad-short"].failed


# ---------------------------------------------------------------- dedup
T1 = "the quick brown fox jumps over the lazy dog near the river bank"
T1_PARA = "the quick brown fox jumps over the lazy dog near the river shore"
T2 = "a completely different sentence about warehouse inventory levels today"


def test_dedup_exact_duplicates_clustered():
    d = Deduplicator()
    d.add("a", "Hello world foo bar baz qux")
    d.add("b", "Hello world foo bar baz qux")
    d.add("c", T2)
    clusters = d.find_duplicates()
    assert clusters == [["a", "b"]]


def test_dedup_near_duplicate_paraphrase_caught():
    d = Deduplicator()
    d.add("a", T1)
    d.add("b", T1_PARA)
    d.add("c", T2)
    clusters = d.find_duplicates()
    assert clusters == [["a", "b"]]


def test_dedup_distinct_items_not_clustered():
    d = Deduplicator()
    d.add("a", T1)
    d.add("b", T2)
    d.add("c", "yet another unrelated sentence about supplier lead times here")
    assert d.find_duplicates() == []


def test_dedup_query_returns_scored_match():
    d = Deduplicator()
    d.add("a", T1)
    d.add("b", T2)
    hits = d.query(T1_PARA)
    assert hits and hits[0][0] == "a"
    assert 0.5 <= hits[0][1] <= 1.0
    assert d.query("zzzz totally novel wording never seen before qqqq") == []


def test_dedup_empty_text_handling():
    d = Deduplicator()
    d.add("e1", "")
    d.add("e2", "   ")
    d.add("f", "nonempty text here")
    clusters = d.find_duplicates()
    assert clusters == [["e1", "e2"]]
    assert d.query("") == []


def test_dedup_deterministic_across_rebuilds():
    def build():
        d = Deduplicator(seed=999)
        d.add("a", T1)
        d.add("b", T1_PARA)
        d.add("c", T2)
        d.add("d", T2)
        return d.find_duplicates()
    assert build() == build() == [["a", "b"], ["c", "d"]]


def test_dedup_exact_match_is_case_and_space_insensitive():
    d = Deduplicator()
    d.add("a", "  HELLO   World  ")
    assert d.exact_match("hello world") == "a"
    assert d.exact_match("hello there") is None


# ---------------------------------------------------------------- decontaminate
def test_decontaminate_removes_seeded_eval_leak():
    train = [
        ("tr-1", "the capital of france is paris and it is lovely"),
        ("tr-2", "photosynthesis converts sunlight into chemical energy now"),
        ("tr-3", "a clean sentence about nothing in particular here today"),
        ("tr-4", "the capital of france is paris and it is beautiful"),
    ]
    eval_items = [
        ("ev-1", "the capital of france is paris and it is lovely"),
    ]
    rep = decontaminate(train, eval_items)
    removed_ids = {r["train_id"] for r in rep.removed}
    assert "tr-1" in removed_ids  # exact leak
    assert "tr-4" in removed_ids  # near-duplicate leak
    assert rep.clean_train_ids == ["tr-2", "tr-3"]
    assert rep.summary["n_removed"] == 2


def test_decontaminate_report_cites_eval_item_and_kind():
    train = [("tr-1", "exact copy of the eval sentence right here now")]
    eval_items = [("ev-9", "exact copy of the eval sentence right here now")]
    rep = decontaminate(train, eval_items)
    assert len(rep.removed) == 1
    r = rep.removed[0]
    assert r["train_id"] == "tr-1" and r["eval_id"] == "ev-9"
    assert r["kind"] == "exact" and r["score"] == 1.0


def test_decontaminate_near_match_scores_below_one():
    train = [("tr-1", T1)]
    eval_items = [("ev-1", T1_PARA)]
    rep = decontaminate(train, eval_items)
    assert len(rep.removed) == 1
    assert rep.removed[0]["kind"] == "near"
    assert 0.5 <= rep.removed[0]["score"] < 1.0


def test_decontaminate_clean_train_untouched():
    train = [("tr-1", T1), ("tr-2", T2)]
    eval_items = [("ev-1", "something entirely unrelated to anything else")]
    rep = decontaminate(train, eval_items)
    assert rep.removed == []
    assert rep.clean_train_ids == ["tr-1", "tr-2"]
    assert rep.summary["n_clean"] == 2


def test_decontaminate_empty_train():
    rep = decontaminate([], [("ev-1", "some eval text")])
    assert rep.removed == [] and rep.clean_train_ids == []
    assert rep.summary["n_removed"] == 0


# ---------------------------------------------------------------- synthesize
POOLS = {
    "tone": ["positive", "critical", "neutral", "mixed"],
    "product": ["headphones", "laptop", "keyboard", "monitor"],
    "topic": ["refund policy", "shipping delays", "warranty terms"],
    "audience": ["new customers", "support agents", "store managers"],
}
TEMPLATES = [
    "Write a {tone} review of the {product} for our store page.",
    "Summarize the {topic} for {audience} in plain language.",
    "Draft a {tone} response to a customer asking about the {topic}.",
]


def test_synthesize_generates_n_items():
    g = SyntheticGenerator(TEMPLATES, POOLS, seed=7)
    items, rejections = g.generate(10)
    assert len(items) == 10
    assert all(i.text and i.item_id for i in items)


def test_synthesize_is_deterministic():
    def run():
        g = SyntheticGenerator(TEMPLATES, POOLS, seed=7)
        items, _ = g.generate(10)
        return [i.text for i in items]
    assert run() == run()


def test_synthesize_no_unfilled_placeholders():
    g = SyntheticGenerator(TEMPLATES, POOLS, seed=7)
    items, _ = g.generate(10)
    for i in items:
        assert "{" not in i.text and "}" not in i.text
        assert (set(i.fields) >= {"tone", "product"}
                or set(i.fields) >= {"topic", "audience"}
                or set(i.fields) >= {"tone", "topic"})


def test_synthesize_length_gate_rejects_with_reason():
    g = SyntheticGenerator(TEMPLATES, POOLS, seed=7, min_chars=500)
    items, rejections = g.generate(5)
    assert items == []
    assert len(rejections) > 0
    assert all(any(r.startswith("length:") for r in rej.reasons)
               for rej in rejections)


def test_synthesize_dedup_gate_rejects_existing():
    g1 = SyntheticGenerator(TEMPLATES, POOLS, seed=7)
    first_batch, _ = g1.generate(3)
    leaked = first_batch[0].text
    g2 = SyntheticGenerator(TEMPLATES, POOLS, seed=7)
    items2, rejections = g2.generate(3, existing_texts=(leaked,))
    dedup_rejects = [r for r in rejections
                     if any("dedup" in reason for reason in r.reasons)]
    assert dedup_rejects, "expected at least one dedup rejection"
    assert all(i.text != leaked for i in items2)


def test_synthesize_diversity_gate_caps_template_share():
    g = SyntheticGenerator(TEMPLATES, POOLS, seed=7, max_template_share=0.4)
    items, _ = g.generate(10)
    counts = {}
    for i in items:
        counts[i.template] = counts.get(i.template, 0) + 1
    assert all(c / len(items) <= 0.4 + 1e-9 for c in counts.values())


def test_synthesize_schema_gate_missing_pool():
    g = SyntheticGenerator(["Hello {name}, welcome!"], {}, seed=1)
    items, rejections = g.generate(3)
    assert items == []
    assert rejections
    assert all(any(r.startswith("schema:") for r in rej.reasons)
               for rej in rejections)


# ---------------------------------------------------------------- cards
def _manifest():
    return {
        "name": "support-sft-v3",
        "version": "3.1.0",
        "source": "synthetic + annotated support conversations",
        "created": "2026-10-18",
        "pipeline_steps": [
            {"name": "filter", "version": "0.1.0",
             "params": {"min_chars": 20}},
            {"name": "dedup", "version": "0.1.0",
             "params": {"jaccard_threshold": 0.5}},
            {"name": "decontaminate", "version": "0.1.0"},
        ],
        "stats": {"n_items": 1250, "n_rejected": 87,
                  "by_label": {"positive": 500, "neutral": 750}},
        "limitations": ["synthetic distribution differs from production",
                        "English only"],
        "license": "internal research use only",
        "contact": "data-eng@example.com",
        "intended_use": "SFT training and offline evals.",
    }


def test_card_renders_all_sections():
    md = build_dataset_card(_manifest())
    for section in ["# Dataset Card: support-sft-v3", "## Pipeline",
                    "## Statistics", "## Known limitations",
                    "## Intended use", "internal research use only"]:
        assert section in md


def test_card_includes_pipeline_versions_and_stats():
    md = build_dataset_card(_manifest())
    assert "filter** (v0.1.0)" in md
    assert "n_items" in md and "1250" in md
    assert "positive: 500" in md


def test_card_lists_limitations():
    md = build_dataset_card(_manifest())
    assert "English only" in md


def test_card_missing_key_raises():
    bad = _manifest()
    del bad["stats"]
    with pytest.raises(ValueError) as exc:
        build_dataset_card(bad)
    assert "stats" in str(exc.value)


def test_card_optional_fields_have_sane_defaults():
    m = _manifest()
    del m["limitations"]
    del m["license"]
    md = build_dataset_card(m)
    assert "unspecified — confirm before use" in md
