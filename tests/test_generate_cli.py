from dataclasses import replace

import pytest

from science_course.cli_config import load_sft_generation_config
from science_course.generate_common import prepare_generation_manifest


def _temporary_config(root, tmp_path):
    config = load_sft_generation_config(root / "configs" / "generate_sft.toml")
    output = replace(
        config.output,
        name="test_generation",
        raw_sources_path=str(tmp_path / "raw.jsonl"),
        accepted_path=str(tmp_path / "accepted.jsonl"),
        rejected_path=str(tmp_path / "rejected.jsonl"),
        dataset_directory=str(tmp_path / "dataset"),
        parquet_directory=str(tmp_path / "parquet"),
        parquet_prefix="test_generation",
        manifest_path=str(tmp_path / "manifest.json"),
    )
    return replace(config, output=output)


def test_manifest_allows_attempt_budget_increase_but_rejects_contract_change(
    tmp_path,
):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    config = _temporary_config(root, tmp_path)
    prepare_generation_manifest(config, stage="generate-sft")

    increased = replace(
        config,
        source=replace(config.source, candidate_limit=3000),
        generation=replace(config.generation, max_attempts=3000),
    )
    prepare_generation_manifest(increased, stage="generate-sft")

    incompatible = replace(
        increased,
        models=replace(increased.models, teacher_model="different-model"),
    )
    with pytest.raises(RuntimeError, match="contract differs"):
        prepare_generation_manifest(incompatible, stage="generate-sft")


def test_manifest_prevents_target_decrease(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    config = _temporary_config(root, tmp_path)
    prepare_generation_manifest(config, stage="generate-sft")

    decreased = replace(
        config,
        targets=replace(config.targets, train=config.targets.train - 1),
    )
    with pytest.raises(RuntimeError, match="cannot be decreased"):
        prepare_generation_manifest(decreased, stage="generate-sft")
