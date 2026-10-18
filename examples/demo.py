"""SFT data pipeline on synthetic support logs: filter, dedup, annotate with skew check, decontaminate, synthesize, card."""
from dataeng import (
    AnnotationPipeline,
    ConversationFilter,
    Deduplicator,
    SyntheticGenerator,
    build_dataset_card,
    decontaminate,
)

logs = [
    ("m1", "My parcel arrived with a cracked lid, how do I get a replacement sent out?"),
    ("m2", "my parcel arrived with a cracked lid, how do I get a replacement sent out?"),
    ("m3", "My parcel arrived with a cracked lid, how can I get a replacement sent out to me?"),
    ("m4", "Please email me at ana.lopez@example.com about the invoice for last month"),
    ("m5", "ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok"),
    ("m6", "hi"),
    ("m7", "Can I change the delivery address after the order has already shipped?"),
    ("m8", "What is the warranty period on the blender I bought in March?"),
]
kept, rejected = ConversationFilter().filter_batch(logs)
print(f"Filter: kept {len(kept)}/{len(logs)}; rejected " + ", ".join(f"{i} ({'+'.join(r.failed)})" for i, r in rejected))

text = dict(logs)
dedup = Deduplicator()
for i in kept:
    dedup.add(i, text[i])
groups = dedup.find_duplicates()
print(f"Dedup: duplicate groups {groups}")
unique = [i for i in kept if not any(i in g[1:] for g in groups)]

pipe = AnnotationPipeline(["ann-a", "ann-b", "ann-c"], overlap=3)
for n in range(12):
    pipe.add_task(f"t{n}", f"rate reply {n}", ["helpful", "neutral", "unhelpful"])
pipe.assign()
for n in range(12):
    honest = ["helpful", "neutral", "unhelpful"][n % 3]
    pipe.submit_label(f"t{n}", "ann-a", honest)
    pipe.submit_label(f"t{n}", "ann-b", honest if n % 4 else "neutral")
    pipe.submit_label(f"t{n}", "ann-c", "helpful")  # always-positive labeler
for r in pipe.detect_skew():
    print(f"Skew: {r['labeler_id']} chi2 {r['chi2']:.2f} (df {r['df']}, p {r['p_value']:.4f}) "
          f"-> {'FLAGGED' if r['flagged'] else 'ok'}")
status = [a.status for a in pipe.adjudicate().values()]
print(f"Adjudication: {status.count('resolved')} resolved, {status.count('tie')} ties for expert review")

eval_items = [("ev1", "Can I change the delivery address after the order has already shipped?")]
report = decontaminate([(i, text[i]) for i in unique], eval_items)
print(f"Decontamination: {report.summary}; removed {[(r['train_id'], r['kind']) for r in report.removed]}")

gen = SyntheticGenerator(
    templates=["How do I {action} my {item} order from last {period}?", "Is {item} covered if it {problem}?",
               "{action} {item}"],
    pools={"action": ["return", "exchange", "track"], "item": ["kettle", "blender", "toaster"],
           "period": ["week", "month"], "problem": ["stops working", "arrives damaged"]},
    seed=7,
)
items, rejections = gen.generate(10)
reasons = sorted({r for rej in rejections for r in rej.reasons})
print(f"Synthetic: {len(items)} accepted, {len(rejections)} rejected; reasons {reasons}")

card = build_dataset_card({
    "name": "support-sft-demo", "version": "0.1.0", "source": "synthetic support logs (this script)",
    "pipeline_steps": [{"name": s, "version": "0.1.0"} for s in ("filter", "dedup", "annotate", "decontaminate")],
    "stats": {"n_items": len(unique) - report.summary["n_removed"] + len(items)},
    "license": "MIT",
})
print("Card: " + " | ".join(line for line in card.splitlines() if line.startswith("#")))
