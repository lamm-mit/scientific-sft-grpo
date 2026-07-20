from dataclasses import replace
from pathlib import Path

import pytest

from science_course.cli import main
from science_course.cli_config import (
    GRPOJobConfig,
    SFTJobConfig,
    latest_checkpoint,
    load_grpo_config,
    load_sft_config,
    resolve_resume,
    with_cli_overrides,
)

ROOT = Path(__file__).resolve().parents[1]


def test_committed_sft_config_matches_notebook_contract():
    config = load_sft_config(ROOT / "configs" / "sft.toml")

    assert isinstance(config, SFTJobConfig)
    assert config.model.model_id == "google/gemma-4-E4B-it"
    assert config.data.hub_config == "scientific_design_sft"
    assert config.lora.rank == 16
    assert config.lora.alpha == 32
    assert config.training.max_sequence_length == 1024
    assert config.training.gradient_accumulation_steps == 8
    assert config.hub.repo_id == "lamm-mit/scientific-sft-grpo-design-sft"
    assert config.output.resume == "auto"


def test_committed_grpo_config_matches_notebook_contract():
    config = load_grpo_config(ROOT / "configs" / "grpo.toml")

    assert isinstance(config, GRPOJobConfig)
    assert config.data.hub_config == "scientific_design_grpo"
    assert config.adapter.hub_repo == "lamm-mit/scientific-sft-grpo-design-sft"
    assert config.reward.judge_model == "gpt-5.6-luna"
    assert config.reward.format_base_reward == pytest.approx(0.10)
    assert config.reward.semantic_reward_weight == pytest.approx(0.90)
    assert config.training.num_generations == 4
    assert config.training.num_generations_eval == 4
    assert config.training.eval_strategy == "epoch"
    assert config.training.eval_steps is None
    assert config.training.loss_type == "dapo"
    assert config.hub.repo_id == "lamm-mit/scientific-sft-grpo-design-grpo"


def test_latest_checkpoint_is_numeric_and_ignores_invalid_directories(tmp_path):
    (tmp_path / "checkpoint-9").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-final").mkdir()
    (tmp_path / "checkpoint-250").write_text("not a directory", encoding="utf-8")

    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-100"
    assert resolve_resume(tmp_path, "auto") == str(tmp_path / "checkpoint-100")
    assert resolve_resume(tmp_path, "none") is None


def test_explicit_missing_resume_checkpoint_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_resume(tmp_path, str(tmp_path / "checkpoint-10"))


def test_smoke_override_is_isolated_and_never_pushes():
    config = load_grpo_config(ROOT / "configs" / "grpo.toml")
    smoke = with_cli_overrides(
        config,
        resume=None,
        push=None,
        smoke_test=True,
    )

    assert smoke.output.directory.endswith("-smoke")
    assert smoke.output.resume == "none"
    assert smoke.training.max_steps == 1
    assert smoke.hub.push is False
    assert config.hub.push is True


def test_reward_coefficients_must_sum_to_one():
    config = load_grpo_config(ROOT / "configs" / "grpo.toml")
    with pytest.raises(ValueError, match="sum to one"):
        replace(
            config.reward,
            format_base_reward=0.2,
            semantic_reward_weight=0.9,
        )


def test_cli_show_config_has_no_credentials(capsys):
    result = main(
        [
            "show-config",
            "--stage",
            "grpo",
            "--config",
            str(ROOT / "configs" / "grpo.toml"),
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert '"judge_model": "gpt-5.6-luna"' in captured.out
    assert "OPENAI_API_KEY" not in captured.out
    assert "HF_TOKEN" not in captured.out
