from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskFamily = Literal[
    "mechanism_guided_design",
    "experimental_design",
    "troubleshooting",
    "hypothesis_development",
    "cross_domain_synthesis",
]


class ScientificDesignTaskDraft(BaseModel):
    """Teacher-created scientific problem-solving task and reference work product."""

    model_config = ConfigDict(extra="forbid")

    usable: bool = Field(
        description=(
            "Whether the source can inspire a scientifically supported, self-contained "
            "problem-solving task."
        )
    )
    rejection_reason: str = Field(
        description="Empty when usable; otherwise why the source cannot support the task."
    )
    task_family: TaskFamily | None = Field(
        description="Null when unusable; otherwise the requested task family."
    )
    task: str | None = Field(
        description=(
            "Null when unusable. Otherwise, a self-contained scientific task containing "
            "all scenario facts and constraints needed without the source document. "
            "Strictly 300–1,600 characters."
        )
    )
    brainstorm: list[str] | None = Field(
        min_length=3,
        max_length=5,
        description=(
            "Null when unusable. Otherwise, three to five distinct, scientifically "
            "plausible candidate approaches or hypotheses. Each item must be 60–240 "
            "characters."
        ),
    )
    principles: list[str] | None = Field(
        min_length=3,
        max_length=6,
        description=(
            "Null when unusable. Otherwise, scientific constraints, design principles, "
            "or evaluation criteria that should guide the solution. Each item must be "
            "40–200 characters."
        ),
    )
    synthesis: str | None = Field(
        description=(
            "Null when unusable. Otherwise, a concise comparison or integration of the "
            "candidates using the stated principles. Strictly 150–750 characters."
        )
    )
    answer: str | None = Field(
        description=(
            "Null when unusable. Otherwise, a decisive final proposal or conclusion in "
            "one compact paragraph of strictly 120–600 characters."
        )
    )
    required_constraints: list[str] | None = Field(
        min_length=2,
        max_length=8,
        description="Task constraints that a strong response must respect.",
    )
    evaluation_criteria: list[str] | None = Field(
        min_length=2,
        max_length=8,
        description="Scientific criteria the judge should use to assess solution quality.",
    )
    acceptable_alternatives: list[str] | None = Field(
        min_length=1,
        max_length=6,
        description=(
            "Scientifically defensible alternative solution directions; the reference "
            "completion is not the only valid answer."
        ),
    )
    failure_modes: list[str] | None = Field(
        min_length=1,
        max_length=6,
        description="Common scientific errors, contradictions, or infeasible approaches.",
    )


class ScientificDesignTaskReview(BaseModel):
    """Independent critic decision for a generated scientific design task."""

    model_config = ConfigDict(extra="forbid")

    accept: bool
    task_is_self_contained: bool
    task_does_not_reference_source: bool
    task_is_qualitative_not_numerical: bool
    brainstorm_is_diverse_and_plausible: bool
    principles_are_scientifically_supported: bool
    synthesis_is_coherent: bool
    answer_is_supported: bool
    rubric_allows_valid_alternatives: bool
    critique: str


class ScientificDesignJudgment(BaseModel):
    """Simple 0–4 semantic grading dimensions for one GRPO completion."""

    model_config = ConfigDict(extra="forbid")

    item_index: int
    brainstorm: int = Field(ge=0, le=4)
    principles: int = Field(ge=0, le=4)
    synthesis: int = Field(ge=0, le=4)
    answer: int = Field(ge=0, le=4)
    brief_reason: str


class ScientificDesignJudgmentBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgments: list[ScientificDesignJudgment]
