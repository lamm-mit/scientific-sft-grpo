from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from .rewards import (
    combined_reward,
    completion_text,
    format_reward,
    semantic_score_from_judgment,
)
from .schemas import ScientificDesignJudgmentBatch
from .teacher import parse_structured_response, require_openai_key, response_usage

DEFAULT_JUDGE_MODEL = "gpt-5.6-luna"
JUDGE_PROMPT_VERSION = "scientific-design-judge-v2"

JUDGE_SYSTEM = """You are a strict but open-minded grader of scientific problem solving.

For every item, score the student's structured response independently on four dimensions.
Use integer scores from 0 to 4:
0 = missing, irrelevant, or scientifically wrong
1 = weak, with major omissions or errors
2 = adequate but incomplete
3 = strong and mostly complete
4 = excellent, coherent, and scientifically defensible

Dimensions:
- brainstorm: distinct, relevant, scientifically plausible candidate ideas;
- principles: appropriate scientific constraints and design principles;
- synthesis: candidates are compared or combined coherently using the principles;
- answer: the final proposal is scientifically sound and directly answers the task.

Use the hidden rubric as criteria, not as a phrase-matching answer key. Accept scientifically
valid alternatives. Penalize unsupported claims, violated constraints, contradictions,
trivial variations presented as different ideas, and synthesis that does not support the
answer. Return exactly one judgment for every item_index."""


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


class ScientificDesignJudge:
    """Batched, cached Luna judge for the four scientific work-product sections."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        cache_path: str | Path = "results/scientific_design_judge_cache.jsonl",
        api_max_retries: int = 3,
        api_timeout_seconds: float = 120.0,
        client: OpenAI | None = None,
    ) -> None:
        require_openai_key()
        self.model = model
        self.cache_path = Path(cache_path)
        if api_max_retries < 0 or api_timeout_seconds <= 0:
            raise ValueError("OpenAI retry and timeout settings must be non-negative.")
        self.client = client or OpenAI(
            max_retries=api_max_retries,
            timeout=api_timeout_seconds,
        )
        self.cache = _load_cache(self.cache_path)

    def judgments(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                            "Grade this JSON batch. Score each response independently and "
                            "allow valid alternatives.\n\n"
                            + json.dumps(judge_items, ensure_ascii=False)
                        ),
                    },
                ],
                text_format=ScientificDesignJudgmentBatch,
                store=False,
            )
            parsed = parse_structured_response(
                response,
                ScientificDesignJudgmentBatch,
            )
            judgments = {item.item_index: item for item in parsed.judgments}
            expected = set(missing_indices)
            if set(judgments) != expected or len(parsed.judgments) != len(expected):
                raise RuntimeError(
                    "Scientific design judge returned unexpected item indexes: "
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

        return [dict(self.cache[key]["judgment"]) for key in keys]

    def score(self, items: list[dict[str, Any]]) -> list[float]:
        return [
            semantic_score_from_judgment(judgment)
            for judgment in self.judgments(items)
        ]


_DEFAULT_JUDGE: ScientificDesignJudge | None = None
_FORMAT_BASE_REWARD = 0.10
_SEMANTIC_REWARD_WEIGHT = 0.90


def configure_scientific_design_judge(
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    cache_path: str | Path = "results/scientific_design_judge_cache.jsonl",
    format_base_reward: float = 0.10,
    semantic_reward_weight: float = 0.90,
    api_max_retries: int = 3,
    api_timeout_seconds: float = 120.0,
    client: OpenAI | None = None,
) -> ScientificDesignJudge:
    """Configure the singleton used by the TRL-compatible combined reward."""

    combined_reward(
        1.0,
        0.0,
        format_base_reward=format_base_reward,
        semantic_reward_weight=semantic_reward_weight,
    )
    global _DEFAULT_JUDGE, _FORMAT_BASE_REWARD, _SEMANTIC_REWARD_WEIGHT
    _DEFAULT_JUDGE = ScientificDesignJudge(
        model=model,
        cache_path=cache_path,
        api_max_retries=api_max_retries,
        api_timeout_seconds=api_timeout_seconds,
        client=client,
    )
    _FORMAT_BASE_REWARD = format_base_reward
    _SEMANTIC_REWARD_WEIGHT = semantic_reward_weight
    return _DEFAULT_JUDGE


def scientific_design_reward(
    completions: list[Any],
    task: list[str],
    required_constraints: list[list[str]],
    evaluation_criteria: list[list[str]],
    acceptable_alternatives: list[list[str]],
    failure_modes: list[list[str]],
    **_: Any,
) -> list[float]:
    """Compute F × (0.10 + 0.90J), skipping API calls for malformed outputs."""

    global _DEFAULT_JUDGE
    if _DEFAULT_JUDGE is None:
        _DEFAULT_JUDGE = ScientificDesignJudge()

    format_scores = format_reward(completions)
    valid_indices = [
        index for index, score in enumerate(format_scores) if score == 1.0
    ]
    items = [
        {
            "task": task[index],
            "required_constraints": required_constraints[index],
            "evaluation_criteria": evaluation_criteria[index],
            "acceptable_alternatives": acceptable_alternatives[index],
            "failure_modes": failure_modes[index],
            "student_response": completion_text(completions[index]),
        }
        for index in valid_indices
    ]
    semantic_scores = _DEFAULT_JUDGE.score(items) if items else []
    semantic_by_index = dict(zip(valid_indices, semantic_scores, strict=True))
    return [
        combined_reward(
            format_score,
            semantic_by_index.get(index, 0.0),
            format_base_reward=_FORMAT_BASE_REWARD,
            semantic_reward_weight=_SEMANTIC_REWARD_WEIGHT,
        )
        for index, format_score in enumerate(format_scores)
    ]
