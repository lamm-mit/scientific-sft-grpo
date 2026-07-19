from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CausalLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause: str
    effect: str


class ScientificTaskDraft(BaseModel):
    """Teacher-created, self-contained scientific mechanism task."""

    model_config = ConfigDict(extra="forbid")

    usable: bool = Field(
        description="Whether the source contains one clear empirical scientific finding."
    )
    rejection_reason: str = Field(
        description="Empty when usable; otherwise why the source cannot support a task."
    )
    task: str | None = Field(
        description=(
            "Null when unusable. Otherwise, a self-contained how/why scientific mechanism "
            "task with all facts needed to answer without the source material."
        )
    )
    reasoning: str | None = Field(
        description=(
            "Null when unusable. Otherwise, a brief visible causal explanation ordered "
            "from cause to outcome."
        )
    )
    evidence: str | None = Field(
        description=(
            "Null when unusable. Otherwise, a short exact quotation of the key observation "
            "as it appears inside the task."
        )
    )
    answer: str | None = Field(
        description="Null when unusable. Otherwise, a concise mechanistic answer to the task."
    )
    mechanism_steps: list[str] | None = Field(
        min_length=2,
        max_length=6,
        description=(
            "Null when unusable. Otherwise, ordered causal steps a correct explanation "
            "should contain."
        ),
    )
    causal_links: list[CausalLink] | None = Field(
        min_length=1,
        max_length=5,
        description=(
            "Null when unusable. Otherwise, cause-effect relations required by the "
            "reference mechanism."
        ),
    )
    required_concepts: list[str] | None = Field(
        min_length=2,
        max_length=8,
        description=(
            "Null when unusable. Otherwise, short concepts a correct paraphrase should preserve."
        ),
    )


class ScientificTaskReview(BaseModel):
    """Independent critic decision for a generated task."""

    model_config = ConfigDict(extra="forbid")

    accept: bool
    task_is_self_contained: bool
    task_is_mechanistic_not_numerical: bool
    evidence_is_quoted_from_task: bool
    answer_is_supported: bool
    reasoning_is_supported: bool
    causal_chain_is_sound: bool
    critique: str


class MechanismJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_index: int
    causal_correctness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    evidence_use: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    concise_feedback: str


class MechanismJudgmentBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgments: list[MechanismJudgment]
