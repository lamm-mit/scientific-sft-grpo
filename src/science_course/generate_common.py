from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from datasets import DatasetDict

from .cli_config import (
    GRPOGenerationConfig,
    SFTGenerationConfig,
    config_as_dict,
)
from .data import (
    read_jsonl,
    stream_open_science_sources,
    task_quota_status,
    write_jsonl,
)
from .hub import require_hf_namespace
from .teacher import PROMPT_VERSION, generate_canonical_tasks, require_openai_key
from .train_common import hf_token, write_json

GenerationConfig = SFTGenerationConfig | GRPOGenerationConfig


def _immutable_contract(config: GenerationConfig, *, stage: str) -> dict[str, Any]:
    source = config.source
    contract: dict[str, Any] = {
        "stage": stage,
        "prompt_version": PROMPT_VERSION,
        "models": {
            "teacher_model": config.models.teacher_model,
            "critic_model": config.models.critic_model,
        },
        "task_family_weights": config.task_family_weights,
        "source": {
            "dataset_id": source.dataset_id,
            "split": source.split,
            "min_chars": source.min_chars,
            "max_chars": source.max_chars,
            "seed": source.seed,
        },
        "journal": {
            "output_name": config.output.name,
            "raw_sources_path": config.output.raw_sources_path,
            "accepted_path": config.output.accepted_path,
            "rejected_path": config.output.rejected_path,
        },
    }
    if isinstance(config, GRPOGenerationConfig):
        contract["sft_canonical_path"] = config.input.sft_canonical_path
    return contract


def _validate_existing_journal(config: GenerationConfig) -> None:
    output = config.output
    rows = [
        *read_jsonl(output.accepted_path),
        *read_jsonl(output.rejected_path),
    ]
    if not rows:
        return
    expected_splits = set(config.targets.split_targets)
    for row in rows:
        row_prompt_version = row.get("prompt_version")
        if row_prompt_version and row_prompt_version != PROMPT_VERSION:
            raise RuntimeError(
                f"Existing journal uses prompt version {row_prompt_version!r}, "
                f"not {PROMPT_VERSION!r}. Choose a new output name and paths."
            )
        if row.get("source_dataset") not in {None, config.source.dataset_id}:
            raise RuntimeError("Existing journal uses a different source dataset.")
        if row.get("source_split") not in {None, config.source.split}:
            raise RuntimeError("Existing journal uses a different source split.")
        split = row.get("split") or row.get("requested_split")
        if split and split not in expected_splits:
            raise RuntimeError(
                f"Existing journal contains incompatible split {split!r}."
            )
        family = row.get("task_family") or row.get("requested_task_family")
        if family and family not in config.task_family_weights:
            raise RuntimeError(
                f"Existing journal contains incompatible task family {family!r}."
            )
        teacher = row.get("teacher")
        if (
            isinstance(teacher, dict)
            and teacher.get("model")
            and teacher["model"] != config.models.teacher_model
        ):
            raise RuntimeError("Existing journal uses a different teacher model.")
        critic = row.get("critic")
        if (
            isinstance(critic, dict)
            and critic.get("model")
            and critic["model"] != config.models.critic_model
        ):
            raise RuntimeError("Existing journal uses a different critic model.")


def prepare_generation_manifest(
    config: GenerationConfig,
    *,
    stage: str,
) -> None:
    """Prevent incompatible runs from appending to the same JSONL journals."""

    manifest_path = Path(config.output.manifest_path)
    contract = _immutable_contract(config, stage=stage)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("immutable_contract") != contract:
            raise RuntimeError(
                f"Generation contract differs from {manifest_path}. "
                "Use a new output name and paths instead of mixing runs."
            )
        old_targets = previous.get("config", {}).get("targets", {})
        new_targets = config_as_dict(config)["targets"]
        decreased = {
            key: (old_targets[key], value)
            for key, value in new_targets.items()
            if key in old_targets and value < old_targets[key]
        }
        if decreased:
            raise RuntimeError(
                "Accepted targets cannot be decreased for an existing journal: "
                f"{decreased}. Use a new output name."
            )
    else:
        _validate_existing_journal(config)
    write_json(
        manifest_path,
        {
            "status": "in_progress",
            "immutable_contract": contract,
            "config": config_as_dict(config),
            "credentials": {
                "openai_api_key_present": True,
                "hf_token_environment": bool(hf_token()),
            },
        },
    )


def acquire_sources(
    config: GenerationConfig,
    *,
    exclude_paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    output = config.output
    source = config.source
    excluded = exclude_paper_ids or set()
    cached = read_jsonl(output.raw_sources_path)
    cache_is_usable = (
        len(cached) == source.candidate_limit
        and not ({str(row["paper_id"]) for row in cached} & excluded)
        and all(
            row.get("source_dataset") == source.dataset_id
            and row.get("source_split") == source.split
            for row in cached
        )
    )
    if cache_is_usable:
        print(
            f"Reusing {len(cached):,} cached sources from "
            f"{output.raw_sources_path}",
            flush=True,
        )
        return cached

    sources = stream_open_science_sources(
        dataset_id=source.dataset_id,
        split=source.split,
        max_papers=source.candidate_limit,
        max_scanned=source.max_records_scanned,
        min_chars=source.min_chars,
        max_chars=source.max_chars,
        seed=source.seed,
        exclude_paper_ids=excluded,
    )
    write_jsonl(output.raw_sources_path, sources)
    print(
        f"Cached {len(sources):,} sources at {output.raw_sources_path}",
        flush=True,
    )
    return sources


def generate_canonical(
    config: GenerationConfig,
    sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = config.targets.split_targets
    target_total = sum(targets.values())
    if len(sources) < target_total:
        raise RuntimeError(
            f"Source pool has {len(sources):,} records but accepted quotas require "
            f"{target_total:,}. Increase source.candidate_limit."
        )
    canonical = generate_canonical_tasks(
        sources,
        accepted_path=config.output.accepted_path,
        rejected_path=config.output.rejected_path,
        split_targets=targets,
        task_family_weights=config.task_family_weights,
        teacher_model=config.models.teacher_model,
        critic_model=config.models.critic_model,
        concurrency=config.generation.concurrency,
        max_attempts=config.generation.max_attempts,
        api_max_retries=config.generation.openai_max_retries,
        api_timeout_seconds=config.generation.openai_timeout_seconds,
    )
    status = task_quota_status(
        canonical,
        targets,
        config.task_family_weights,
    )
    if not status["complete"]:
        raise RuntimeError(f"Unfilled accepted-example quotas: {status['deficits']}")
    return canonical, status


def save_projection(
    dataset: DatasetDict,
    config: GenerationConfig,
    *,
    expected_splits: dict[str, int],
    commit_message: str,
) -> dict[str, Any]:
    actual_splits = {name: len(split) for name, split in dataset.items()}
    if actual_splits != expected_splits:
        raise RuntimeError(
            f"Unexpected split sizes: {actual_splits}; expected {expected_splits}"
        )

    destination = Path(config.output.dataset_directory).expanduser()
    resolved = destination.resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in protected:
        raise RuntimeError(f"Refusing to replace unsafe dataset path: {resolved}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(destination)

    parquet_directory = Path(config.output.parquet_directory).expanduser()
    parquet_directory.mkdir(parents=True, exist_ok=True)
    parquet_paths = {}
    for split_name, split_data in dataset.items():
        parquet_path = (
            parquet_directory
            / f"{config.output.parquet_prefix}_{split_name}.parquet"
        )
        split_data.to_parquet(parquet_path)
        parquet_paths[split_name] = str(parquet_path)

    hub_result = None
    token = hf_token()
    if config.hub.push:
        hub_result = dataset.push_to_hub(
            config.hub.repo_id,
            config_name=config.hub.config_name,
            private=config.hub.private,
            token=token,
            commit_message=commit_message,
        )
    return {
        "splits": actual_splits,
        "dataset_directory": str(destination),
        "parquet": parquet_paths,
        "hub_repository": config.hub.repo_id if config.hub.push else None,
        "hub_config": config.hub.config_name if config.hub.push else None,
        "hub_result": str(hub_result) if hub_result is not None else None,
    }


def generation_preflight(config: GenerationConfig) -> None:
    require_openai_key()
    if config.hub.push:
        require_hf_namespace(config.hub.repo_id, token=hf_token())


def complete_generation_manifest(
    config: GenerationConfig,
    *,
    stage: str,
    result: dict[str, Any],
) -> None:
    write_json(
        config.output.manifest_path,
        {
            "status": "complete",
            "immutable_contract": _immutable_contract(config, stage=stage),
            "config": config_as_dict(config),
            "result": result,
        },
    )
