from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

ALLOWED_LICENSES = {
    "CC0",
    "CC00",
    "CCBY",
    "CCBY4",
    "CCBY40",
    "CCBYSA",
    "CCBYSA4",
    "CCBYSA40",
    "PUBLICDOMAIN",
    "PD",
}

TASK_FAMILIES = (
    "mechanism_guided_design",
    "experimental_design",
    "troubleshooting",
    "hypothesis_development",
    "cross_domain_synthesis",
)

SYSTEM_TASK = (
    "Solve the self-contained scientific problem-solving task. Develop several distinct "
    "candidate ideas, identify the governing scientific principles and constraints, "
    "synthesize the strongest direction, and give a concise final answer."
)

RESPONSE_INSTRUCTIONS = """Respond in exactly this order, with no text outside the tags:
<brainstorm>
three to five distinct candidate ideas
</brainstorm>
<principles>
the scientific constraints and design principles
</principles>
<synthesis>
compare or combine the candidates using the principles
</synthesis>
<answer>
a concise final proposal or conclusion
</answer>"""


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_license(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata", {})
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def allocate_task_quotas(
    split_targets: Mapping[str, int],
    task_family_weights: Mapping[str, float],
) -> dict[tuple[str, str], int]:
    """Allocate exact integer family quotas inside each requested split."""

    if not split_targets or any(int(value) < 0 for value in split_targets.values()):
        raise ValueError("Split targets must be a non-empty mapping of non-negative counts.")
    if set(task_family_weights) != set(TASK_FAMILIES):
        raise ValueError(f"Task-family weights must define exactly: {TASK_FAMILIES}")
    if any(float(value) < 0 for value in task_family_weights.values()):
        raise ValueError("Task-family weights cannot be negative.")
    weight_total = sum(float(value) for value in task_family_weights.values())
    if weight_total <= 0:
        raise ValueError("At least one task-family weight must be positive.")

    allocation: dict[tuple[str, str], int] = {}
    for split, target_value in split_targets.items():
        target = int(target_value)
        raw = {
            family: target * float(weight) / weight_total
            for family, weight in task_family_weights.items()
        }
        counts = {family: math.floor(value) for family, value in raw.items()}
        remainder = target - sum(counts.values())
        ranked = sorted(
            TASK_FAMILIES,
            key=lambda family: (-(raw[family] - counts[family]), family),
        )
        for family in ranked[:remainder]:
            counts[family] += 1
        for family in TASK_FAMILIES:
            allocation[(str(split), family)] = counts[family]
    return allocation


def task_quota_status(
    records: Iterable[dict[str, Any]],
    split_targets: Mapping[str, int],
    task_family_weights: Mapping[str, float],
) -> dict[str, Any]:
    targets = allocate_task_quotas(split_targets, task_family_weights)
    observed = Counter(
        (str(record.get("split", "")), str(record.get("task_family", "")))
        for record in records
    )
    deficits = {
        key: max(target - observed[key], 0)
        for key, target in targets.items()
        if observed[key] < target
    }
    return {
        "target_total": sum(targets.values()),
        "accepted_total": sum(
            min(observed[key], target) for key, target in targets.items()
        ),
        "complete": not deficits,
        "targets": targets,
        "observed": dict(observed),
        "deficits": deficits,
    }


def trim_records_to_quotas(
    records: Iterable[dict[str, Any]],
    split_targets: Mapping[str, int],
    task_family_weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Select a deterministic, exact quota-conforming subset of accepted records."""

    targets = allocate_task_quotas(split_targets, task_family_weights)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {
        key: [] for key in targets
    }
    for record in records:
        key = (str(record.get("split", "")), str(record.get("task_family", "")))
        if key in grouped:
            grouped[key].append(record)

    selected: list[dict[str, Any]] = []
    for key, target in targets.items():
        rows = sorted(
            grouped[key],
            key=lambda record: (
                str(record.get("task_id", "")),
                str(record.get("paper_id", "")),
            ),
        )
        selected.extend(rows[:target])
    return sorted(selected, key=lambda record: str(record.get("task_id", "")))


def _source_excerpt(text: str, *, min_chars: int, max_chars: int) -> str | None:
    paragraphs = [normalize_whitespace(p) for p in re.split(r"\n\s*\n", text)]
    paragraphs = [p for p in paragraphs if len(p) >= 80]
    if not paragraphs:
        return None

    selected: list[str] = []
    total = 0
    for paragraph in paragraphs:
        if selected and total + len(paragraph) + 2 > max_chars:
            break
        selected.append(paragraph)
        total += len(paragraph) + 2
    source_text = "\n\n".join(selected)
    return source_text if len(source_text) >= min_chars else None


def stream_open_science_sources(
    *,
    dataset_id: str = "common-pile/peS2o",
    split: str = "train",
    max_papers: int = 2_500,
    max_scanned: int = 200_000,
    min_chars: int = 1_200,
    max_chars: int = 6_000,
    seed: int = 17,
    exclude_paper_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Stream a license-filtered scientific source pool for task authoring."""

    stream = load_dataset(dataset_id, split=split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=min(max_scanned, 20_000))
    records: list[dict[str, Any]] = []
    excluded = {str(value) for value in exclude_paper_ids}
    seen = set(excluded)

    for index, row in enumerate(stream):
        if index >= max_scanned or len(records) >= max_papers:
            break
        meta = _metadata(row)
        license_raw = (
            meta.get("oa_license")
            or meta.get("license")
            or row.get("license")
            or ""
        )
        license_normalized = normalize_license(license_raw)
        if license_normalized not in ALLOWED_LICENSES:
            continue

        source = str(row.get("source", ""))
        if source and "s2orc" not in source.lower():
            continue

        paper_id = str(row.get("id") or row.get("paper_id") or f"row-{index}")
        if paper_id in seen:
            continue
        source_text = _source_excerpt(
            str(row.get("text") or row.get("raw_fulltext") or ""),
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if source_text is None:
            continue

        title = str(meta.get("title") or row.get("title") or paper_id)
        records.append(
            {
                "paper_id": paper_id,
                "title": normalize_whitespace(title)[:300],
                "source_text": source_text,
                "source_dataset": dataset_id,
                "source_split": split,
                "source_url": str(meta.get("oa_url") or meta.get("url") or ""),
                "source_license": str(license_raw),
                "source_license_normalized": license_normalized,
                "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            }
        )
        seen.add(paper_id)
    return records


def task_prompt(task: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_TASK},
        {
            "role": "user",
            "content": (
                "SCIENTIFIC PROBLEM-SOLVING TASK\n"
                f"{task}\n\n"
                f"{RESPONSE_INSTRUCTIONS}"
            ),
        },
    ]


def _render_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_completion(record: dict[str, Any]) -> str:
    return (
        f"<brainstorm>\n{_render_list(record['brainstorm'])}\n</brainstorm>\n"
        f"<principles>\n{_render_list(record['principles'])}\n</principles>\n"
        f"<synthesis>\n{record['synthesis']}\n</synthesis>\n"
        f"<answer>\n{record['answer']}\n</answer>"
    )


def canonical_to_sft(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": record["task_id"],
        "paper_id": record["paper_id"],
        "task_family": record["task_family"],
        "prompt": task_prompt(record["task"]),
        "completion": [{"role": "assistant", "content": render_completion(record)}],
        "source_license": record["source_license"],
        "source_url": record["source_url"],
    }


def canonical_to_grpo(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": record["task_id"],
        "paper_id": record["paper_id"],
        "task_family": record["task_family"],
        "prompt": task_prompt(record["task"]),
        "task": record["task"],
        "required_constraints": record["required_constraints"],
        "evaluation_criteria": record["evaluation_criteria"],
        "acceptable_alternatives": record["acceptable_alternatives"],
        "failure_modes": record["failure_modes"],
        "source_license": record["source_license"],
        "source_url": record["source_url"],
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def build_sft_dataset(records: list[dict[str, Any]]) -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_list(
                [canonical_to_sft(r) for r in records if r["split"] == "sft_train"]
            ),
            "validation": Dataset.from_list(
                [
                    canonical_to_sft(r)
                    for r in records
                    if r["split"] == "sft_validation"
                ]
            ),
        }
    )


def build_grpo_dataset(records: list[dict[str, Any]]) -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_list(
                [canonical_to_grpo(r) for r in records if r["split"] == "grpo_train"]
            ),
            "validation": Dataset.from_list(
                [
                    canonical_to_grpo(r)
                    for r in records
                    if r["split"] == "grpo_validation"
                ]
            ),
            "test": Dataset.from_list(
                [canonical_to_grpo(r) for r in records if r["split"] == "test"]
            ),
        }
    )
