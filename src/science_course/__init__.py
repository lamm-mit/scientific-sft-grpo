"""Teaching utilities for scientific SFT and GRPO notebooks."""

from .devices import RuntimeDevice, detect_runtime
from .hub import require_hf_namespace
from .judge import (
    ScientificDesignJudge,
    configure_scientific_design_judge,
    scientific_design_reward,
)
from .rewards import combined_reward, format_reward, parse_completion
from .schemas import ScientificDesignTaskDraft, ScientificDesignTaskReview

__all__ = [
    "RuntimeDevice",
    "ScientificDesignJudge",
    "ScientificDesignTaskDraft",
    "ScientificDesignTaskReview",
    "combined_reward",
    "configure_scientific_design_judge",
    "detect_runtime",
    "format_reward",
    "parse_completion",
    "require_hf_namespace",
    "scientific_design_reward",
]
