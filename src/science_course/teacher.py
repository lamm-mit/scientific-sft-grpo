from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from .data import (
    TASK_FAMILIES,
    allocate_task_quotas,
    normalize_whitespace,
    read_jsonl,
    task_quota_status,
    trim_records_to_quotas,
)
from .schemas import ScientificDesignTaskDraft, ScientificDesignTaskReview

DEFAULT_TEACHER_MODEL = "gpt-5.6-terra"
DEFAULT_CRITIC_MODEL = "gpt-5.6-terra"
PROMPT_VERSION = "scientific-problem-solving-v3"

SchemaT = TypeVar("SchemaT", bound=BaseModel)

TASK_FAMILY_GUIDANCE = {
    "mechanism_guided_design": (
        "Ask the student to design or improve a scientific system by using mechanisms "
        "supported by the source."
    ),
    "experimental_design": (
        "Ask the student to propose experiments that test a mechanism, distinguish "
        "hypotheses, or diagnose a causal process."
    ),
    "troubleshooting": (
        "Present a plausible scientific failure or unexpected behavior and ask the student "
        "to develop and synthesize mechanistic remedies."
    ),
    "hypothesis_development": (
        "Present an observation and ask the student to develop several mechanistic "
        "hypotheses, identify discriminating principles, and select a leading explanation."
    ),
    "cross_domain_synthesis": (
        "Ask the student to transfer or combine scientific principles across systems into "
        "a defensible design or research direction."
    ),
}

TEACHER_SYSTEM = """You create high-quality scientific problem-solving tasks from source text.

The student will see only the task, never the source. Use the source as scientific grounding
for one NEW, SELF-CONTAINED task. Put all scenario facts and constraints needed to solve the
task directly in the task. Never mention a passage, paper, source, article, or unavailable
material. Focus on qualitative mechanisms, scientific design, hypotheses, experimentation,
troubleshooting, and synthesis—not arithmetic or retrieval.

Create a reference work product with:
1. three to five distinct candidate ideas in brainstorm;
2. scientific constraints and design principles;
3. a synthesis that compares or combines candidates using those principles; and
4. a concise, decisive answer.

HARD LENGTH LIMITS (count characters, including spaces):
- task: 300–1,600 characters;
- each brainstorm idea: 60–240 characters;
- each principle: 40–200 characters;
- synthesis: 150–750 characters;
- answer: 120–600 characters;
- each hidden-rubric item: 30–180 characters.
Treat these as strict output requirements. Prefer one compact paragraph for synthesis and
one compact paragraph for answer. Do not turn the task or answer into a long protocol.

Also create a hidden grading rubric containing required constraints, evaluation criteria,
acceptable alternative approaches, and failure modes. The reference is one strong solution,
not the only valid solution. Never introduce scientific claims unsupported by the source.
When unusable, set task_family and every task, response, and rubric field to null."""

CRITIC_SYSTEM = """You are a strict scientific dataset reviewer.

Compare the proposed task and reference work product with the source text. Accept only when:
- the task stands alone and never refers to the source;
- the task is qualitative rather than a calculation exercise;
- brainstorming contains genuinely distinct, scientifically plausible directions;
- principles and constraints are supported;
- synthesis uses the stated principles rather than merely repeating ideas;
- the final answer is supported and responsive; and
- the hidden rubric permits scientifically defensible alternatives.

Reject answer leakage, unsupported claims, trivial variations presented as diverse ideas,
incoherent synthesis, or a rubric that rewards only phrase matching."""


def require_openai_key() -> str:
    token = os.environ.get("OPENAI_API_KEY", "").strip()
    if not token:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Export it before running dataset generation."
        )
    return token


def parse_structured_response(response: Any, schema: type[SchemaT]) -> SchemaT:
    direct = getattr(response, "output_parsed", None)
    if isinstance(direct, schema):
        return direct
    for output in getattr(response, "output", []):
        if getattr(output, "type", None) != "message":
            continue
        for item in getattr(output, "content", []):
            refusal = getattr(item, "refusal", None)
            if refusal:
                raise RuntimeError(f"OpenAI refused the dataset-generation request: {refusal}")
            parsed = getattr(item, "parsed", None)
            if isinstance(parsed, schema):
                return parsed
    raise RuntimeError("The Responses API returned no parsed structured output.")


def response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    result = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = int(value)
    return result


def _call(
    client: OpenAI,
    *,
    model: str,
    schema: type[SchemaT],
    system: str,
    user: str,
) -> tuple[SchemaT, dict[str, Any]]:
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=schema,
        store=False,
    )
    return parse_structured_response(response, schema), {
        "response_id": getattr(response, "id", None),
        "model": getattr(response, "model", model),
        "usage": response_usage(response),
    }


def _duplicates(values: list[str]) -> bool:
    normalized = [normalize_whitespace(value).casefold() for value in values]
    return len(set(normalized)) != len(normalized)


def validate_draft(
    draft: ScientificDesignTaskDraft,
    *,
    requested_family: str,
) -> list[str]:
    errors = []
    if not draft.usable:
        errors.append(draft.rejection_reason or "teacher marked source unusable")
        return errors

    if draft.task_family != requested_family:
        errors.append(
            f"teacher returned task family {draft.task_family!r}; expected {requested_family!r}"
        )

    task = draft.task or ""
    brainstorm = list(draft.brainstorm or [])
    principles = list(draft.principles or [])
    synthesis = draft.synthesis or ""
    answer = draft.answer or ""
    constraints = list(draft.required_constraints or [])
    criteria = list(draft.evaluation_criteria or [])
    alternatives = list(draft.acceptable_alternatives or [])
    failure_modes = list(draft.failure_modes or [])
    task_normalized = normalize_whitespace(task).casefold()

    unavailable_source_phrases = (
        "according to the passage",
        "according to the source",
        "in the source text",
        "in the paper",
        "the article states",
    )
    if any(phrase in task_normalized for phrase in unavailable_source_phrases):
        errors.append("task refers to unavailable source material")

    numerical_instructions = (
        "calculate",
        "compute",
        "estimate the value",
        "determine the value",
        "what is the value",
        "how many",
    )
    if any(instruction in task_normalized for instruction in numerical_instructions):
        errors.append("task asks for a numerical result")

    problem_solving_cues = (
        "design",
        "develop",
        "propose",
        "hypoth",
        "experiment",
        "diagnos",
        "troubleshoot",
        "synthesi",
        "explain",
    )
    if not any(cue in task_normalized for cue in problem_solving_cues):
        errors.append("task does not clearly request scientific problem solving")

    if not 120 <= len(task) <= 2_000:
        errors.append("task length is outside 120..2000 characters")
    if not 40 <= len(synthesis) <= 1_200:
        errors.append("synthesis length is outside 40..1200 characters")
    if not 30 <= len(answer) <= 700:
        errors.append("answer length is outside 30..700 characters")
    if not 3 <= len(brainstorm) <= 5:
        errors.append("brainstorm must contain three to five ideas")
    if not 3 <= len(principles) <= 6:
        errors.append("principles must contain three to six items")

    named_lists = {
        "brainstorm": brainstorm,
        "principles": principles,
        "required constraints": constraints,
        "evaluation criteria": criteria,
        "acceptable alternatives": alternatives,
        "failure modes": failure_modes,
    }
    for name, values in named_lists.items():
        if not values:
            errors.append(f"{name} is empty")
        elif _duplicates(values):
            errors.append(f"{name} contains duplicate items")
    return errors


def generate_task(
    source_record: dict[str, Any],
    *,
    split: str,
    task_family: str,
    client: OpenAI,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if task_family not in TASK_FAMILIES:
        raise ValueError(f"Unknown task family: {task_family}")

    source_text = source_record["source_text"]
    draft, teacher_meta = _call(
        client,
        model=teacher_model,
        schema=ScientificDesignTaskDraft,
        system=TEACHER_SYSTEM,
        user=(
            f"REQUESTED TASK FAMILY\n{task_family}\n\n"
            f"FAMILY GUIDANCE\n{TASK_FAMILY_GUIDANCE[task_family]}\n\n"
            f"SCIENTIFIC SOURCE TEXT\n\n{source_text}"
        ),
    )
    errors = validate_draft(draft, requested_family=task_family)
    request_fields = {"requested_split": split, "requested_task_family": task_family}
    if errors:
        return None, {
            **source_record,
            **request_fields,
            "stage": "teacher_validation",
            "prompt_version": PROMPT_VERSION,
            "reasons": errors,
            "teacher": teacher_meta,
            "draft": draft.model_dump(),
        }

    review, critic_meta = _call(
        client,
        model=critic_model,
        schema=ScientificDesignTaskReview,
        system=CRITIC_SYSTEM,
        user=(
            f"SCIENTIFIC SOURCE TEXT\n\n{source_text}\n\n"
            f"PROPOSED TASK\n\n{draft.model_dump_json(indent=2)}"
        ),
    )
    review_flags = (
        review.accept,
        review.task_is_self_contained,
        review.task_does_not_reference_source,
        review.task_is_qualitative_not_numerical,
        review.brainstorm_is_diverse_and_plausible,
        review.principles_are_scientifically_supported,
        review.synthesis_is_coherent,
        review.answer_is_supported,
        review.rubric_allows_valid_alternatives,
    )
    if not all(review_flags):
        return None, {
            **source_record,
            **request_fields,
            "stage": "critic",
            "prompt_version": PROMPT_VERSION,
            "reasons": [review.critique or "critic rejected task"],
            "teacher": teacher_meta,
            "critic": critic_meta,
            "draft": draft.model_dump(),
            "review": review.model_dump(),
        }

    task_id = hashlib.sha256(
        (
            f"{PROMPT_VERSION}:{split}:{task_family}:"
            f"{source_record['paper_id']}:{source_record['source_sha256']}"
        ).encode()
    ).hexdigest()[:20]
    accepted = {
        "task_id": task_id,
        **source_record,
        "split": split,
        **draft.model_dump(
            exclude={"usable", "rejection_reason", "task_family"}
        ),
        "task_family": task_family,
        "teacher": teacher_meta,
        "critic": critic_meta,
        "review": review.model_dump(),
        "prompt_version": PROMPT_VERSION,
    }
    return accepted, None


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _planned_slots(
    accepted: list[dict[str, Any]],
    targets: dict[tuple[str, str], int],
    limit: int,
) -> list[tuple[str, str]]:
    counts = Counter(
        (str(row.get("split", "")), str(row.get("task_family", "")))
        for row in accepted
    )
    ranked_units: list[tuple[float, str, str, int]] = []
    for (split, family), target in targets.items():
        current = min(counts[(split, family)], target)
        for ordinal in range(current, target):
            ranked_units.append((ordinal / max(target, 1), split, family, ordinal))
    ranked_units.sort()
    return [(split, family) for _, split, family, _ in ranked_units[:limit]]


def generate_canonical_tasks(
    sources: list[dict[str, Any]],
    *,
    accepted_path: str | Path,
    rejected_path: str | Path,
    split_targets: Mapping[str, int],
    task_family_weights: Mapping[str, float],
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
    concurrency: int = 8,
    max_attempts: int | None = None,
    api_max_retries: int = 3,
    api_timeout_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    """Fill exact post-validation quotas with resumable concurrent API generation."""

    require_openai_key()
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if api_max_retries < 0 or api_timeout_seconds <= 0:
        raise ValueError("OpenAI retry and timeout settings must be non-negative.")

    accepted_path = Path(accepted_path)
    rejected_path = Path(rejected_path)
    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path)
    targets = allocate_task_quotas(split_targets, task_family_weights)

    source_ids = {str(row["paper_id"]) for row in sources}
    completed = {
        str(row["paper_id"])
        for row in [*accepted, *rejected]
        if str(row.get("paper_id", "")) in source_ids
    }
    remaining = [row for row in sources if str(row["paper_id"]) not in completed]
    if max_attempts is not None:
        remaining_budget = max(int(max_attempts) - len(completed), 0)
        remaining = remaining[:remaining_budget]

    client = OpenAI(
        max_retries=api_max_retries,
        timeout=api_timeout_seconds,
    )
    initial = task_quota_status(accepted, split_targets, task_family_weights)
    progress = tqdm(
        total=initial["target_total"],
        initial=initial["accepted_total"],
        desc=f"Accepted scientific tasks ({teacher_model})",
    )

    source_index = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while source_index < len(remaining):
            status = task_quota_status(accepted, split_targets, task_family_weights)
            if status["complete"]:
                break

            batch_size = min(concurrency, len(remaining) - source_index)
            slots = _planned_slots(accepted, targets, batch_size)
            batch = remaining[source_index : source_index + len(slots)]
            source_index += len(batch)
            futures = {
                executor.submit(
                    generate_task,
                    source,
                    split=slot[0],
                    task_family=slot[1],
                    client=client,
                    teacher_model=teacher_model,
                    critic_model=critic_model,
                ): (source, slot)
                for source, slot in zip(batch, slots, strict=True)
            }
            for future in as_completed(futures):
                source, slot = futures[future]
                try:
                    task, rejection = future.result()
                except Exception as error:
                    task = None
                    rejection = {
                        **source,
                        "requested_split": slot[0],
                        "requested_task_family": slot[1],
                        "stage": "api_error",
                        "prompt_version": PROMPT_VERSION,
                        "reasons": [f"{type(error).__name__}: {error}"],
                    }
                if task is not None:
                    _append_jsonl(accepted_path, task)
                    accepted.append(task)
                    progress.update(1)
                elif rejection is not None:
                    _append_jsonl(rejected_path, rejection)
                    rejected.append(rejection)
    progress.close()

    status = task_quota_status(accepted, split_targets, task_family_weights)
    if not status["complete"]:
        readable = {
            f"{split}/{family}": count
            for (split, family), count in status["deficits"].items()
        }
        raise RuntimeError(
            "Generation stopped before filling the requested accepted-task quotas. "
            f"Remaining deficits: {readable}. Increase the source pool or attempt budget "
            "and rerun; completed work is already cached."
        )
    return trim_records_to_quotas(
        accepted,
        split_targets,
        task_family_weights,
    )
