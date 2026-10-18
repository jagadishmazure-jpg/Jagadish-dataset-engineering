# Jagadish-dataset-engineering: data pipelines for SFT and eval quality.
"""Annotation, filtering, dedup, decontamination, synthesis, and dataset cards."""

from .annotate import AnnotationPipeline, Label, Task, AdjudicationResult
from .filters import ConversationFilter, FilterResult
from .dedup import Deduplicator
from .decontaminate import decontaminate, DecontaminationReport
from .synthesize import SyntheticGenerator, GeneratedItem, Rejection
from .cards import build_dataset_card

__all__ = [
    "AnnotationPipeline",
    "Label",
    "Task",
    "AdjudicationResult",
    "ConversationFilter",
    "FilterResult",
    "Deduplicator",
    "decontaminate",
    "DecontaminationReport",
    "SyntheticGenerator",
    "GeneratedItem",
    "Rejection",
    "build_dataset_card",
]

__version__ = "0.1.0"
