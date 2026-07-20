from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from accelerate import PartialState
from huggingface_hub import HfApi

from .cli_config import GRPOJobConfig, SFTJobConfig, resolve_resume
from .devices import detect_runtime
from .hub import require_hf_namespace
from .train_common import (
    configure_environment,
    hf_token,
    load_training_dataset,
    require_dataset_schema,
)
from .train_grpo import GRPO_RUBRIC_COLUMNS
from .versions import require_training_stack


def _check(
    results: list[dict[str, Any]],
    name: str,
    operation: Callable[[], Any],
) -> Any | None:
    try:
        detail = operation()
    except Exception as error:
        results.append(
            {
                "name": name,
                "status": "failed",
                "detail": f"{type(error).__name__}: {error}",
            }
        )
        return None
    results.append({"name": name, "status": "passed", "detail": detail})
    return detail


def _model_access(model_id: str, token: str | None) -> dict[str, Any]:
    info = HfApi(token=token).model_info(model_id)
    return {
        "model_id": info.id,
        "gated": getattr(info, "gated", None),
        "private": info.private,
    }


def _adapter_access(config: GRPOJobConfig, token: str | None) -> dict[str, Any]:
    if config.adapter.source == "local":
        source = Path(config.adapter.local_path).expanduser()
        if not source.is_dir():
            raise FileNotFoundError(f"Local adapter not found: {source}")
        if not (source / "adapter_config.json").is_file():
            raise FileNotFoundError(f"adapter_config.json not found in {source}")
        return {"source": "local", "path": str(source)}
    info = HfApi(token=token).model_info(config.adapter.hub_repo)
    return {"source": "hub", "model_id": info.id, "private": info.private}


def _dataset_check(
    config: SFTJobConfig | GRPOJobConfig,
    token: str | None,
    *,
    stage: str,
) -> dict[str, Any]:
    dataset = load_training_dataset(config.data, token=token)
    if stage == "sft":
        columns = {"prompt", "completion"}
    else:
        columns = GRPO_RUBRIC_COLUMNS
    require_dataset_schema(
        dataset,
        splits=("train", "validation"),
        columns=columns,
    )
    return {
        "source": config.data.source,
        "splits": {split: len(rows) for split, rows in dataset.items()},
        "required_columns": sorted(columns),
    }


def run_doctor(
    config: SFTJobConfig | GRPOJobConfig,
    *,
    stage: str,
) -> dict[str, Any]:
    """Perform read-only preflight checks without loading model weights."""

    configure_environment(config.runtime)
    token = hf_token()
    checks: list[dict[str, Any]] = []
    _check(checks, "training_stack", require_training_stack)
    _check(checks, "runtime", lambda: detect_runtime().as_dict())
    _check(
        checks,
        "base_model_access",
        lambda: _model_access(config.model.model_id, token),
    )
    _check(
        checks,
        "dataset_schema",
        lambda: _dataset_check(config, token, stage=stage),
    )
    _check(
        checks,
        "checkpoint_state",
        lambda: {
            "output_directory": config.output.directory,
            "configured_resume": config.output.resume,
            "resolved_resume": resolve_resume(
                config.output.directory,
                config.output.resume,
            ),
        },
    )
    if stage == "grpo":
        assert isinstance(config, GRPOJobConfig)
        _check(
            checks,
            "sft_adapter_access",
            lambda: _adapter_access(config, token),
        )
        _check(
            checks,
            "openai_api_key",
            lambda: (
                {"present": True}
                if os.environ.get("OPENAI_API_KEY")
                else (_ for _ in ()).throw(
                    RuntimeError("OPENAI_API_KEY is not set.")
                )
            ),
        )
        _check(
            checks,
            "single_process",
            lambda: (
                {"world_size": 1}
                if PartialState().num_processes == 1
                else (_ for _ in ()).throw(
                    RuntimeError(
                        "The live API judge requires one process (WORLD_SIZE=1)."
                    )
                )
            ),
        )
    if config.hub.push:
        _check(
            checks,
            "hub_write_access",
            lambda: {
                "identity": require_hf_namespace(
                    config.hub.repo_id,
                    token=token,
                ).get("name"),
                "repository": config.hub.repo_id,
            },
        )
    passed = all(check["status"] == "passed" for check in checks)
    return {
        "stage": stage,
        "passed": passed,
        "credentials": {
            "hf_token_environment": bool(os.environ.get("HF_TOKEN")),
            "hf_cached_auth_allowed": True,
            "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        },
        "checks": checks,
    }
