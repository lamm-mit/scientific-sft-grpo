from __future__ import annotations

import re
from collections import Counter
from typing import Any

TAG_PATTERN = re.compile(
    r"^\s*<reasoning>(?P<reasoning>.*?)</reasoning>\s*"
    r"<evidence>(?P<evidence>.*?)</evidence>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*$",
    flags=re.DOTALL | re.IGNORECASE,
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
    return {key: value.strip() for key, value in match.groupdict().items()}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.casefold())


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(_tokens(prediction))
    expected = Counter(_tokens(reference))
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((predicted & expected).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    return [1.0 if parse_completion(item) is not None else 0.0 for item in completions]


def evidence_grounding_reward(
    completions: list[Any],
    task: list[str],
    **_: Any,
) -> list[float]:
    scores = []
    for completion, source in zip(completions, task, strict=True):
        parsed = parse_completion(completion)
        evidence = parsed["evidence"] if parsed else ""
        grounded = bool(evidence) and _normalize(evidence) in _normalize(source)
        scores.append(1.0 if grounded else 0.0)
    return scores


def concept_coverage_reward(
    completions: list[Any],
    required_concepts: list[list[str]],
    **_: Any,
) -> list[float]:
    scores = []
    for completion, concepts in zip(completions, required_concepts, strict=True):
        parsed = parse_completion(completion)
        combined = (
            f"{parsed['reasoning']} {parsed['answer']}" if parsed is not None else ""
        )
        normalized = _normalize(combined)
        concepts = list(concepts or [])
        covered = sum(_normalize(concept) in normalized for concept in concepts)
        scores.append(covered / len(concepts) if concepts else 0.0)
    return scores


def answer_similarity_reward(
    completions: list[Any],
    reference_answer: list[str],
    **_: Any,
) -> list[float]:
    scores = []
    for completion, reference in zip(completions, reference_answer, strict=True):
        parsed = parse_completion(completion)
        answer = parsed["answer"] if parsed else ""
        scores.append(token_f1(answer, reference))
    return scores


def component_scores(
    completion: Any,
    *,
    task: str,
    reference_answer: str,
    required_concepts: list[str],
) -> dict[str, float]:
    return {
        "format": format_reward([completion])[0],
        "evidence": evidence_grounding_reward([completion], [task])[0],
        "concepts": concept_coverage_reward([completion], [required_concepts])[0],
        "similarity": answer_similarity_reward([completion], [reference_answer])[0],
    }
