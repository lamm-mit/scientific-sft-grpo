"""Teaching utilities for scientific SFT and GRPO notebooks."""

from .devices import RuntimeDevice, detect_runtime
from .hub import require_hf_namespace
from .judge import configure_mechanism_judge, mechanism_judge_reward
from .rewards import (
    answer_similarity_reward,
    concept_coverage_reward,
    evidence_grounding_reward,
    format_reward,
)
from .schemas import ScientificTaskDraft, ScientificTaskReview

__all__ = [
    "RuntimeDevice",
    "ScientificTaskDraft",
    "ScientificTaskReview",
    "answer_similarity_reward",
    "concept_coverage_reward",
    "configure_mechanism_judge",
    "detect_runtime",
    "evidence_grounding_reward",
    "format_reward",
    "mechanism_judge_reward",
    "require_hf_namespace",
]
