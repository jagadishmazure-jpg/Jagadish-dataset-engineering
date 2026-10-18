# Pipeline notes: why data quality bounds SFT

Distilled from the AI Engineering literature (notably Chip Huyen's *AI
Engineering* framing, ch. 8 "Dataset Engineering" and ch. 1's
Product → Data → Model loop). Ideas paraphrased; see the book for the source.

## The core claim

For fine-tuning and evaluation, **data pipelines bound quality before model
choice does**. A better base model cannot recover signal that annotation
bias, duplicates, eval leakage, or un-QC'd synthetic data destroyed. Data
work is therefore not janitorial — it is the highest-leverage engineering in
the SFT loop.

## Annotation QC

- Human labels are the ground truth everything else trains against, so
  labeler quality *is* model quality. Two failure modes dominate:
  - **Disagreement**: genuine ambiguity. Handle with overlap (multiple
    labelers per item) and majority-vote adjudication; persistent ties go to
    expert review, not to coin flips.
  - **Skew**: a labeler's systematic bias (always-positive, severity
    shifting, rushing). Aggregate agreement metrics hide it because the
    skewed labeler agrees with the majority on easy items. Detect it by
    comparing each labeler's label *distribution* against peers
    (chi-square-style) — this repo's `detect_skew()`.
- Comparison data (for RLHF/DPO) deserves the same QC: the preference loss
  is only as honest as the comparisons.

## Decontamination

- Eval leakage — train items that duplicate or near-duplicate eval items —
  is the quietest way to lie with benchmarks. Scores inflate, generalization
  gaps hide, and model selection optimizes for memorization.
- Decontamination must be **exact + near-duplicate** (paraphrases leak too),
  and it must **report** what it removed against which eval item, so the
  cleaning itself is auditable. That is `decontaminate()`.

## Synthetic data

- Synthetic data is a force multiplier with three failure modes: malformed
  items (schema violations), duplicates (of real data and of itself), and
  diversity collapse (one template dominating). All three are cheap to gate
  at generation time and expensive to debug in training curves.
- Every rejection should carry a reason. Silent drops make the generator
  un-debuggable; logged reasons make it tunable.

## The loop

Product → Data → Model: production behavior generates data, data quality
work turns it into training/eval sets, models improve, and the loop
repeats. This repo is the **Data** step made executable. The **Product**
step (collecting feedback) is its sibling project; together they close the
loop the book centers on.

## What this repo implements

| Concept | Module |
|---|---|
| Annotation overlap + adjudication + skew detection | `dataeng/annotate.py` |
| Raw-log heuristic filters | `dataeng/filters.py` |
| MinHash/LSH near-dedup | `dataeng/dedup.py` |
| Eval decontamination with cited removals | `dataeng/decontaminate.py` |
| QC-gated synthetic generation | `dataeng/synthesize.py` |
| Dataset cards from run manifests | `dataeng/cards.py` |
