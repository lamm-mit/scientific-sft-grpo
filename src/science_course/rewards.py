from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

TAG_PATTERN = re.compile(
    r"^\s*<brainstorm>(?P<brainstorm>.*?)</brainstorm>\s*"
    r"<principles>(?P<principles>.*?)</principles>\s*"
    r"<synthesis>(?P<synthesis>.*?)</synthesis>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)

JUDGMENT_DIMENSIONS = ("brainstorm", "principles", "synthesis", "answer")
JUDGE_MAX_SCORE = 4
SECTION_TAG_PATTERN = re.compile(
    r"</?(?:brainstorm|principles|synthesis|answer)>",
    flags=re.IGNORECASE,
)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(completion)


def parse_completion(completion: Any) -> dict[str, str] | None:
    match = TAG_PATTERN.fullmatch(completion_text(completion))
    if match is None:
        return None
    parsed = {key: value.strip() for key, value in match.groupdict().items()}
    if not all(parsed.values()):
        return None
    if any(SECTION_TAG_PATTERN.search(value) for value in parsed.values()):
        return None
    return parsed


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    """Return one only for four non-empty, correctly ordered sections."""

    return [1.0 if parse_completion(item) is not None else 0.0 for item in completions]


def semantic_score_from_judgment(judgment: Mapping[str, Any]) -> float:
    """Normalize four integer 0–4 judge dimensions to [0, 1]."""

    values = [int(judgment[name]) for name in JUDGMENT_DIMENSIONS]
    if any(value < 0 or value > JUDGE_MAX_SCORE for value in values):
        raise ValueError(f"Judge dimensions must be in 0..{JUDGE_MAX_SCORE}.")
    return sum(values) / (len(values) * JUDGE_MAX_SCORE)


def combined_reward(
    format_score: float,
    semantic_score: float,
    *,
    format_base_reward: float = 0.10,
    semantic_reward_weight: float = 0.90,
) -> float:
    """Exact teaching reward: F × (base + semantic_weight × J)."""

    if not 0.0 <= format_score <= 1.0:
        raise ValueError("format_score must be between 0 and 1")
    if not 0.0 <= semantic_score <= 1.0:
        raise ValueError("semantic_score must be between 0 and 1")
    if format_base_reward < 0 or semantic_reward_weight < 0:
        raise ValueError("Reward coefficients cannot be negative")
    if abs(format_base_reward + semantic_reward_weight - 1.0) > 1e-9:
        raise ValueError("format_base_reward and semantic_reward_weight must sum to 1")
    return format_score * (
        format_base_reward + semantic_reward_weight * semantic_score
    )
