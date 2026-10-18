# Interview points

Short, honest talking points for Jagadish-dataset-engineering. Every number quoted is from the
real run shown in the README.

- Data quality caps model quality; I made each step testable so the guarantees are in tests, not in a slide.
- A fixed chi-square threshold ignores degrees of freedom; I report the p-value, and the demo shows a labeler the threshold misses.
- My dedup misses a reworded duplicate at the default threshold; I keep that visible instead of tuning it away.
- Decontamination cites the eval item for every removal, so the step itself can be audited.

## Likely follow-up questions

- How would you scale dedup? (MinHash/LSH is what large Spark jobs use; move shingling out, keep confirmation.)
- How do you pick the skew threshold? (From the p-value and label count, plus a minimum number of labels.)
- Why not only embeddings for dedup? (Cost and explainability; use them for paraphrase on top of MinHash.)
