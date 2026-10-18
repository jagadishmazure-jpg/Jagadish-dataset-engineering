# Jagadish-dataset-engineering: data pipelines for SFT and eval quality.
"""Annotation, filtering, dedup, decontamination, synthesis, and dataset cards."""

from .annotate import AdjudicationResult, AnnotationPipeline, Label, Task
from .cards import build_dataset_card
from .decontaminate import DecontaminationReport, decontaminate
from .dedup import Deduplicator
from .filters import ConversationFilter, FilterResult
from .synthesize import GeneratedItem, Rejection, SyntheticGenerator

__all__ = [
    "AdjudicationResult",
    "AnnotationPipeline",
    "ConversationFilter",
    "DecontaminationReport",
    "Deduplicator",
    "FilterResult",
    "GeneratedItem",
    "Label",
    "Rejection",
    "SyntheticGenerator",
    "Task",
    "build_dataset_card",
    "decontaminate",
]

__version__ = "0.1.0"
