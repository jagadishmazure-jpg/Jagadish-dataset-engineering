# Jagadish-dataset-engineering

Data pipelines for SFT and eval quality — the unglamorous layer that bounds everything above it.

## Problem

Model quality is bounded by data pipelines long before it is bounded by model
choice. Four failure modes silently cap SFT and eval quality in production:

1. **Labeler skew** — one annotator's systematic bias (always-positive,
   severity-shifting) poisons the golden set, and nobody notices because
   aggregate agreement looks fine.
2. **Near-duplicate contamination** — paraphrased copies inflate dataset size
   while teaching the model nothing and leaking across splits.
3. **Eval leakage** — train items that are copies or near-copies of eval items
   inflate benchmark scores and hide real generalization gaps.
4. **Un-QC'd synthetic data** — template-generated items with malformed
   fields, duplicates, or collapsed diversity silently narrow the model.

## Approach

An executable, testable pipeline. Raw conversation logs flow through:

```
raw logs → filter → dedup → annotate/adjudicate → decontaminate → card
                ↘ synthesize (QC-gated) ↗
```

| Module | What it does |
|---|---|
| `dataeng/annotate.py` | Workforce annotation: deterministic task assignment with overlap, label collection, majority-vote adjudication (ties flagged for expert review), chi-square-style labeler skew detection. Every label cites `(task_id, labeler_id)`. |
| `dataeng/filters.py` | Heuristic gates for raw logs: length bounds, language stub, PII patterns (email/phone/SSN/card), repetition detector. Rejections name the failed check. |
| `dataeng/dedup.py` | Near-duplicate detection: hand-rolled MinHash (seeded md5) + LSH banding, exact-dup fast path, Jaccard confirmation. |
| `dataeng/decontaminate.py` | Eval decontamination: strips train items matching eval items (exact + near), with a report citing each removal's eval match and score. |
| `dataeng/synthesize.py` | Template-based synthetic generation with QC gates: schema, length, dedup, diversity share caps. Rejections logged with reasons. |
| `dataeng/cards.py` | Dataset cards: renders a markdown card (source, pipeline steps + versions, stats, limitations, license) from a run manifest. |

## Outcome

40 deterministic tests prove the pipeline's guarantees: skewed labelers get
flagged, paraphrase-level duplicates get caught, seeded eval leaks get fully
removed, malformed synthetic items get rejected with reasons, and every
dataset ships with a card describing how it was built.

## Quickstart

```python
from dataeng import (
    AnnotationPipeline, ConversationFilter, Deduplicator,
    decontaminate, SyntheticGenerator, build_dataset_card,
)

# 1. Filter raw logs
f = ConversationFilter()
kept, rejected = f.filter_batch([("m1", "Hello, I need help with my order")])

# 2. Dedup
d = Deduplicator()
for i, text in enumerate(["the cat sat", "the cat sat", "dogs bark loudly today here"]):
    d.add(f"doc-{i}", text)
print(d.find_duplicates())  # [['doc-0', 'doc-1']]

# 3. Annotate with skew detection
p = AnnotationPipeline(["ann-a", "ann-b", "ann-c"], overlap=2)
p.add_task("t0", "rate this reply", ["positive", "neutral", "negative"])
p.assign()
p.submit_label("t0", "ann-a", "positive")
print(p.detect_skew())

# 4. Decontaminate train against eval
report = decontaminate(train=[("tr1", "...")], eval_items=[("ev1", "...")])
print(report.summary)

# 5. Generate synthetic data behind QC gates
g = SyntheticGenerator(
    templates=["Write a {tone} review of the {product}."],
    pools={"tone": ["positive", "critical"], "product": ["headphones", "laptop"]},
    seed=42,
)
items, rejections = g.generate(100)

# 6. Card the dataset
print(build_dataset_card({
    "name": "support-sft-v3", "version": "3.1.0",
    "source": "annotated support conversations",
    "pipeline_steps": [{"name": "filter", "version": "0.1.0"}],
    "stats": {"n_items": 1250},
}))
```

Run the tests:

```bash
python -m pytest tests/ -q
```

## Production path

- **Annotation**: swap the in-process pipeline for a workforce tool (Label Studio,
  Scale AI, Surge); keep the adjudication + skew-detection math — it ports as-is.
- **Dedup at scale**: MinHash/LSH here is the same algorithm Spark jobs run;
  move shingling to Spark/Dask, keep the Jaccard confirmation threshold.
- **Storage**: emit HF `datasets`-compatible Parquet + the dataset card as
  `README.md` on the Hub; wire decontamination into the training-data build job.
- **Feedback loop**: pair with `Jagadish-agent-feedback-loop` — production
  corrections become newly annotated tasks, closing Product → Data → Model.

## Knowledge

- [knowledge/pipeline-notes.md](knowledge/pipeline-notes.md) — why data quality
  bounds SFT: annotation QC, decontamination, the iteration loop.
- [knowledge/architecture.md](knowledge/architecture.md) — pipeline architecture
  (Mermaid).

---

*All data in this project is synthetic and generated for engineering purposes.
This is an independent project, not affiliated with or endorsed by any employer,
client, or vendor. Tool and product names are used descriptively only.*
