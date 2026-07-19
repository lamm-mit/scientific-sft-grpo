from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from .data import normalize_whitespace, read_jsonl
from .schemas import ScientificTaskDraft, ScientificTaskReview

DEFAULT_TEACHER_MODEL = "gpt-5.6-terra"
PROMPT_VERSION = "mechanism-task-v1"

SchemaT = TypeVar("SchemaT", bound=BaseModel)

TEACHER_SYSTEM = """You create high-quality scientific mechanism tasks from source text.

Use the source only to construct one new SELF-CONTAINED how/why task. A student will see
the task but will not see the source. Put every observation and scientific fact needed to
solve it directly in the task. Do not refer to unavailable source material or ask for recall.
Focus on mechanisms and causal explanation, not arithmetic or numerical calculation.

Create a brief visible pedagogical reasoning chain, an evidence quotation copied exactly
from the self-contained task, and a concise mechanistic answer. Also provide ordered
mechanism steps, explicit cause-effect links, and short required concepts for grading.
If the source cannot support a clear mechanism task, mark usable=false. Never introduce
scientific claims not supported by the source. When usable=false, set every task,
response, and rubric field to null."""

CRITIC_SYSTEM = """You are a strict scientific dataset reviewer.
Check the proposed mechanism task against the source text. The task must stand alone
without the source, must ask how or why rather than request arithmetic, and must contain
all observations needed to solve it. Reject unsupported causal links, answer leakage,
evidence not copied from the task, or reasoning that overclaims the source."""


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


def validate_draft(draft: ScientificTaskDraft) -> list[str]:
    errors = []
    if not draft.usable:
        errors.append(draft.rejection_reason or "teacher marked source unusable")
        return errors
    task = draft.task or ""
    reasoning = draft.reasoning or ""
    evidence = draft.evidence or ""
    answer = draft.answer or ""
    required_concepts = draft.required_concepts or []
    task_normalized = normalize_whitespace(task).casefold()
    if not evidence or normalize_whitespace(evidence).casefold() not in task_normalized:
        errors.append("teacher evidence is not a verbatim normalized task substring")
    unavailable_source_phrases = (
        "according to the passage",
        "according to the source",
        "in the source text",
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
        errors.append("task asks for a numerical result instead of a causal mechanism")
    mechanism_cues = ("explain", "how ", "why ", "mechanism", "causes", "leads to")
    if not any(cue in task_normalized for cue in mechanism_cues):
        errors.append("task does not clearly request a how/why causal explanation")
    if not 80 <= len(task) <= 2_000:
        errors.append("task length is outside 80..2000 characters")
    if not 20 <= len(answer) <= 500:
        errors.append("answer length is outside 20..500 characters")
    if not 20 <= len(reasoning) <= 800:
        errors.append("reasoning length is outside 20..800 characters")
    if len({c.casefold() for c in required_concepts}) != len(required_concepts):
        errors.append("required concepts contain duplicates")
    if (
        not draft.mechanism_steps
        or not draft.causal_links
        or not draft.required_concepts
    ):
        errors.append("task has no explicit causal rubric")
    return errors


def generate_task(
    source_record: dict[str, Any],
    *,
    client: OpenAI,
    model: str = DEFAULT_TEACHER_MODEL,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source_text = source_record["source_text"]
    draft, teacher_meta = _call(
        client,
        model=model,
        schema=ScientificTaskDraft,
        system=TEACHER_SYSTEM,
        user=f"SCIENTIFIC SOURCE TEXT\n\n{source_text}",
    )
    errors = validate_draft(draft)
    if errors:
        return None, {
            **source_record,
            "stage": "teacher_validation",
            "prompt_version": PROMPT_VERSION,
            "reasons": errors,
            "teacher": teacher_meta,
            "draft": draft.model_dump(),
        }

    review, critic_meta = _call(
        client,
        model=model,
        schema=ScientificTaskReview,
        system=CRITIC_SYSTEM,
        user=(
            f"SCIENTIFIC SOURCE TEXT\n\n{source_text}\n\n"
            f"PROPOSED TASK\n\n{draft.model_dump_json(indent=2)}"
        ),
    )
    if not (
        review.accept
        and review.task_is_self_contained
        and review.task_is_mechanistic_not_numerical
        and review.evidence_is_quoted_from_task
        and review.answer_is_supported
        and review.reasoning_is_supported
        and review.causal_chain_is_sound
    ):
        return None, {
            **source_record,
            "stage": "critic",
            "prompt_version": PROMPT_VERSION,
            "reasons": [review.critique or "critic rejected task"],
            "teacher": teacher_meta,
            "critic": critic_meta,
            "draft": draft.model_dump(),
            "review": review.model_dump(),
        }

    task_id = hashlib.sha256(
        f"{PROMPT_VERSION}:{source_record['paper_id']}:{source_record['source_sha256']}".encode()
    ).hexdigest()[:20]
    accepted = {
        "task_id": task_id,
        **source_record,
        **draft.model_dump(exclude={"usable", "rejection_reason"}),
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


def generate_canonical_tasks(
    sources: list[dict[str, Any]],
    *,
    accepted_path: str | Path,
    rejected_path: str | Path,
    model: str = DEFAULT_TEACHER_MODEL,
) -> list[dict[str, Any]]:
    """Generate, critique, and incrementally persist tasks; safe to rerun."""

    require_openai_key()
    accepted_path = Path(accepted_path)
    rejected_path = Path(rejected_path)
    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path)
    completed = {
        row["paper_id"]
        for row in [*accepted, *rejected]
        if "paper_id" in row
    }
    remaining = [row for row in sources if row["paper_id"] not in completed]
    client = OpenAI(max_retries=3, timeout=120.0)

    for source in tqdm(remaining, desc=f"GPT task generation ({model})"):
        task, rejection = generate_task(source, client=client, model=model)
        if task is not None:
            _append_jsonl(accepted_path, task)
            accepted.append(task)
        elif rejection is not None:
            _append_jsonl(rejected_path, rejection)
            rejected.append(rejection)
    return accepted
