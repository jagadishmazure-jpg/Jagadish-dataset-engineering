# Architecture

## Pipeline flow

```mermaid
flowchart LR
    RAW["raw conversation logs"] --> F["filter\nConversationFilter\nlength / language / PII / repetition"]
    F -->|kept| D["dedup\nDeduplicator\nMinHash + LSH\nJaccard confirm"]
    F -->|rejected| RJ1[("rejection log\n(named checks)")]
    D -->|unique| A["annotate\nAnnotationPipeline\nassign → label →\nadjudicate"]
    D -->|duplicates| RJ2[("duplicate clusters")]
    A -->|resolved| DC["decontaminate\nvs eval set\nexact + near"]
    A -->|ties| EXPERT(["expert review queue"])
    A -->|skew flags| SKEW(["labeler skew report"])
    SYN["synthesize\nSyntheticGenerator\ntemplates + pools"] --> QCG{"QC gates\nschema / length /\ndedup / diversity"}
    QCG -->|accepted| DC
    QCG -->|rejected| RJ3[("rejection log\n(reasons)")]
    DC -->|clean train| CARD["card\nbuild_dataset_card\nsource · steps · stats ·\nlimitations · license"]
    DC -->|removed| RJ4[("decontamination report\n(train_id → eval_id, score)")]
```

## Skew detection

```mermaid
flowchart TB
    L["labels\n(task_id, labeler_id, label)"] --> SPLIT["per-labeler\nlabel distributions"]
    SPLIT --> PEER["peer-pooled distribution\n(all *other* labelers +\nLaplace smoothing)"]
    SPLIT --> CHI{"chi-square\n> threshold?"}
    PEER --> CHI
    CHI -->|yes| FLAG(["flag labeler\n(report distribution)"])
    CHI -->|no| OK(["labeler healthy"])
```

## Design rules

- **Deterministic**: seeded RNG (synthesis), seeded md5 MinHash (dedup),
  rotating assignment (annotation). Same inputs → same outputs, always.
- **Cited**: labels cite `(task_id, labeler_id)`; decontamination removals
  cite `(train_id → eval_id, score)`; rejections cite reasons.
- **Stdlib only**: no numpy, no sklearn — every algorithm is hand-rolled and
  readable, which is the point for an engineering portfolio.
- **Gates, not filters**: nothing is silently dropped. Every rejection path
  produces a log entry with a reason.
