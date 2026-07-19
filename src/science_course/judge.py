from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from .rewards import completion_text
from .schemas import MechanismJudgmentBatch
from .teacher import (
    DEFAULT_TEACHER_MODEL,
    parse_structured_response,
    require_openai_key,
    response_usage,
)

JUDGE_PROMPT_VERSION = "mechanism-judge-v1"

JUDGE_SYSTEM = """You are a strict grader of scientific mechanism explanations.

For every item, evaluate the student's response against the self-contained task and
hidden causal rubric. Reward a scientifically correct causal chain: the initiating
condition or perturbation, relevant intermediate processes, and the resulting outcome.
Paraphrases are valid. Do not reward mere keyword overlap, unsupported causal claims,
or a correct final claim reached through incorrect reasoning.

Score causal_correctness, completeness, evidence_use, and overall_score from 0 to 1.
Evidence must be quoted from the task and used consistently with the causal explanation.
The task is qualitative: never require a numerical result. Return exactly one judgment
for each item_index."""


def _cache_key(model: str, item: dict[str, Any]) -> str:
    payload = {
        "prompt_version": JUDGE_PROMPT_VERSION,
        "model": model,
        "item": item,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                cache[row["cache_key"]] = row
    return cache


def _append_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class MechanismJudge:
    """Batched, cached LLM judge used as the semantic GRPO reward."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_TEACHER_MODEL,
        cache_path: str | Path = "results/grpo_judge_cache.jsonl",
        client: OpenAI | None = None,
    ) -> None:
        require_openai_key()
        self.model = model
        self.cache_path = Path(cache_path)
        self.client = client or OpenAI(max_retries=3, timeout=120.0)
        self.cache = _load_cache(self.cache_path)

    def score(self, items: list[dict[str, Any]]) -> list[float]:
        keys = [_cache_key(self.model, item) for item in items]
        missing_indices = [
            index for index, key in enumerate(keys) if key not in self.cache
        ]

        if missing_indices:
            judge_items = [
                {"item_index": index, **items[index]} for index in missing_indices
            ]
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            "Grade this JSON batch. The reference is a rubric, not a phrase "
                            "matching target.\n\n"
                            + json.dumps(judge_items, ensure_ascii=False)
                        ),
                    },
                ],
                text_format=MechanismJudgmentBatch,
                store=False,
            )
            parsed = parse_structured_response(response, MechanismJudgmentBatch)
            judgments = {item.item_index: item for item in parsed.judgments}
            expected = set(missing_indices)
            if set(judgments) != expected or len(parsed.judgments) != len(expected):
                raise RuntimeError(
                    "Mechanism judge returned unexpected item indexes: "
                    f"expected {sorted(expected)}, received {sorted(judgments)}"
                )

            response_meta = {
                "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", self.model),
                "usage": response_usage(response),
            }
            rows = []
            for index in missing_indices:
                judgment = judgments[index].model_dump()
                row = {
                    "cache_key": keys[index],
                    "prompt_version": JUDGE_PROMPT_VERSION,
                    "request": items[index],
                    "judgment": judgment,
                    "response": response_meta,
                }
                self.cache[keys[index]] = row
                rows.append(row)
            _append_cache(self.cache_path, rows)

        return [
            float(self.cache[key]["judgment"]["overall_score"])
            for key in keys
        ]


_DEFAULT_JUDGE: MechanismJudge | None = None


def configure_mechanism_judge(
    *,
    model: str = DEFAULT_TEACHER_MODEL,
    cache_path: str | Path = "results/grpo_judge_cache.jsonl",
    client: OpenAI | None = None,
) -> MechanismJudge:
    """Configure the singleton used by the TRL-compatible judge reward."""

    global _DEFAULT_JUDGE
    _DEFAULT_JUDGE = MechanismJudge(
        model=model,
        cache_path=cache_path,
        client=client,
    )
    return _DEFAULT_JUDGE


def mechanism_judge_reward(
    completions: list[Any],
    task: list[str],
    reference_reasoning: list[str],
    reference_answer: list[str],
    mechanism_steps: list[list[str]],
    causal_links: list[list[dict[str, str]]],
    **_: Any,
) -> list[float]:
    """TRL-compatible semantic reward; one cached API request per reward batch."""

    global _DEFAULT_JUDGE
    if _DEFAULT_JUDGE is None:
        _DEFAULT_JUDGE = MechanismJudge()

    items = [
        {
            "task": current_task,
            "reference_reasoning": current_reasoning,
            "reference_answer": current_answer,
            "mechanism_steps": current_steps,
            "causal_links": current_links,
            "student_response": completion_text(completion),
        }
        for (
            completion,
            current_task,
            current_reasoning,
            current_answer,
            current_steps,
            current_links,
        ) in zip(
            completions,
            task,
            reference_reasoning,
            reference_answer,
            mechanism_steps,
            causal_links,
            strict=True,
        )
    ]
    return _DEFAULT_JUDGE.score(items)
