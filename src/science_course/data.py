from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
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

SYSTEM_TASK = (
    "Solve the self-contained scientific mechanism task. Explain the causal chain, quote "
    "the most relevant observation already included in the task, and give a concise answer."
)

SPLIT_THRESHOLDS = (
    ("sft_train", 50),
    ("sft_validation", 60),
    ("grpo_train", 80),
    ("grpo_validation", 90),
    ("test", 100),
)


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


def assign_paper_split(paper_id: str, seed: int = 17) -> str:
    digest = hashlib.sha256(f"{seed}:{paper_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    for name, upper in SPLIT_THRESHOLDS:
        if bucket < upper:
            return name
    raise AssertionError(bucket)


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
    max_papers: int = 120,
    max_scanned: int = 20_000,
    min_chars: int = 1_200,
    max_chars: int = 6_000,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Stream a small license-filtered classroom sample of scientific text."""

    stream = load_dataset(dataset_id, split=split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=min(max_scanned, 10_000))
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

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
                "source_url": str(meta.get("oa_url") or meta.get("url") or ""),
                "source_license": str(license_raw),
                "source_license_normalized": license_normalized,
                "split": assign_paper_split(paper_id, seed),
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
                "SCIENTIFIC MECHANISM TASK\n"
                f"{task}\n\n"
                "Respond in exactly this order:\n"
                "<reasoning>brief visible rationale</reasoning>\n"
                "<evidence>exact observation quoted from the task</evidence>\n"
                "<answer>concise mechanistic answer</answer>"
            ),
        },
    ]


def render_completion(record: dict[str, Any]) -> str:
    return (
        f"<reasoning>{record['reasoning']}</reasoning>\n"
        f"<evidence>{record['evidence']}</evidence>\n"
        f"<answer>{record['answer']}</answer>"
    )


def canonical_to_sft(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": record["task_id"],
        "paper_id": record["paper_id"],
        "prompt": task_prompt(record["task"]),
        "completion": [{"role": "assistant", "content": render_completion(record)}],
        "source_license": record["source_license"],
        "source_url": record["source_url"],
    }


def canonical_to_grpo(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": record["task_id"],
        "paper_id": record["paper_id"],
        "prompt": task_prompt(record["task"]),
        "task": record["task"],
        "reference_reasoning": record["reasoning"],
        "reference_answer": record["answer"],
        "reference_evidence": record["evidence"],
        "mechanism_steps": record["mechanism_steps"],
        "causal_links": record["causal_links"],
        "required_concepts": record["required_concepts"],
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
