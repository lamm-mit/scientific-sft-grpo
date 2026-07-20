from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset, load_from_disk

from .cli_config import DatasetOptions, RuntimeOptions


def configure_environment(options: RuntimeOptions) -> None:
    if options.enable_mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = str(
        options.tokenizers_parallelism
    ).lower()


def hf_token() -> str | None:
    """Use an explicit environment token when present, otherwise cached Hub auth."""

    return os.environ.get("HF_TOKEN") or None


def load_training_dataset(
    options: DatasetOptions,
    *,
    token: str | None,
) -> DatasetDict:
    if options.source == "hub":
        return load_dataset(
            options.hub_repo,
            options.hub_config,
            token=token,
        )
    source = Path(options.local_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Local dataset does not exist: {source}")
    dataset = load_from_disk(source)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {source}; found {type(dataset).__name__}.")
    return dataset


def require_dataset_schema(
    dataset: DatasetDict,
    *,
    splits: tuple[str, ...],
    columns: set[str],
) -> None:
    missing_splits = [split for split in splits if split not in dataset]
    if missing_splits:
        raise ValueError(f"Dataset is missing splits: {', '.join(missing_splits)}")
    for split in splits:
        if len(dataset[split]) == 0:
            raise ValueError(f"Dataset split '{split}' is empty.")
        missing_columns = columns - set(dataset[split].column_names)
        if missing_columns:
            raise ValueError(
                f"Dataset split '{split}' is missing columns: "
                + ", ".join(sorted(missing_columns))
            )


def smoke_subset(dataset: DatasetDict) -> DatasetDict:
    limits = {"train": 8, "validation": 4, "test": 1}
    return DatasetDict(
        {
            split: rows.select(range(min(len(rows), limits.get(split, 1))))
            for split, rows in dataset.items()
        }
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    temporary.replace(destination)


def run_manifest(
    *,
    stage: str,
    config: Any,
    runtime: Any,
    versions: dict[str, str],
    resume_from_checkpoint: str | None,
    smoke_test: bool,
) -> dict[str, Any]:
    """Return a secret-free, reproducible record of the resolved run."""

    return {
        "stage": stage,
        "config": asdict(config),
        "runtime": runtime.as_dict(),
        "versions": versions,
        "resume_from_checkpoint": resume_from_checkpoint,
        "smoke_test": smoke_test,
        "credentials": {
            "hf_token_environment": bool(os.environ.get("HF_TOKEN")),
            "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        },
    }
