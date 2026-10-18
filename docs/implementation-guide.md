# Implementation guide: from offline build to Azure

> Status: **planned**. Nothing here has been deployed. The offline build in this
> repository is what exists today; this guide is how it would be taken to a
> real Azure environment.

## Azure service mapping

| Offline component | Azure service (planned) | Notes |
|---|---|---|
| Raw logs | Fabric lakehouse bronze tables | PII filter before silver |
| Dedup and decontamination | Spark notebooks in Fabric or Azure Databricks with the same shingling | Keep the Jaccard confirmation |
| Annotation | A labelling tool with the adjudication and skew maths kept in code | Labeler ids pseudonymised |
| Cards | Card stored next to the dataset version | Versioned with the data |

## Steps

1. Run the filters on ingestion and keep the rejection reasons as a table.
2. Move shingling and MinHash to Spark; keep the thresholds and the confirmation step.
3. Set the skew rule from the p-value and the number of label options, not a fixed statistic.
4. Block training jobs whose dataset card is missing a decontamination step.

## Guardrails to keep when going live

- PII patterns reject a log before it can reach a dataset.
- Every filter, dedup, decontamination and synthesis decision is reported with its reason or score.
- Ties in adjudication go to expert review instead of being broken arbitrarily.
- Eval decontamination runs before the card is written, and the card records the pipeline steps and versions.

## What to measure before calling it production-ready

- Rejection rate per filter over time (a jump usually means an upstream change).
- Duplicate and eval-leak rates per dataset version.
- Inter-annotator agreement and the number of flagged labelers per batch.
