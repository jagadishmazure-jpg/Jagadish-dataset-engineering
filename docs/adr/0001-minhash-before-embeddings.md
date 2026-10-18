# ADR 0001: MinHash/LSH dedup first, embeddings later

Status: accepted (2026-10-18)

## Context
Duplicates come in two kinds: near-copies (case, whitespace, small edits) and
paraphrases. MinHash over word shingles is cheap, deterministic and
explainable, since every match comes with a Jaccard score. It misses
paraphrases.

## Decision
Use an exact fast path, then MinHash with LSH banding, then confirm with exact
Jaccard. Keep paraphrase detection as a planned embedding stage on top.

## Consequences
- Runs offline with no model and gives the same result every run.
- The demo shows a known miss: a reworded duplicate (m3) passes at the default
  threshold of 0.5. It stays visible rather than being hidden by tuning.
