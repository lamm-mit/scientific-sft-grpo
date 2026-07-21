from dataclasses import replace
from pathlib import Path

import pytest

from science_course.cli import main
from science_course.cli_config import (
    GRPOJobConfig,
    SFTJobConfig,
    latest_checkpoint,
    load_grpo_config,
    load_grpo_generation_config,
    load_sft_config,
    load_sft_generation_config,
    resolve_resume,
    with_cli_overrides,
    with_generation_push_override,
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


def test_committed_generation_configs_match_notebook_contract():
    sft = load_sft_generation_config(ROOT / "configs" / "generate_sft.toml")
    grpo = load_grpo_generation_config(ROOT / "configs" / "generate_grpo.toml")

    assert sft.targets.split_targets == {
        "sft_train": 500,
        "sft_validation": 75,
    }
    assert sft.source.candidate_limit == 2500
    assert sft.output.accepted_path.endswith("scientific_design_sft_tasks.jsonl")
    assert sft.hub.config_name == "scientific_design_sft"
    assert grpo.targets.split_targets == {
        "grpo_train": 500,
        "grpo_validation": 75,
        "test": 100,
    }
    assert grpo.input.sft_canonical_path == (
        "data/canonical/scientific_design_sft_tasks.jsonl"
    )
    assert grpo.source.seed == 29
    assert grpo.hub.config_name == "scientific_design_grpo"


def test_large_generation_configs_have_exact_requested_totals_and_isolated_paths():
    sft = load_sft_generation_config(ROOT / "configs" / "generate_sft_L.toml")
    grpo = load_grpo_generation_config(ROOT / "configs" / "generate_grpo_L.toml")

    assert sum(sft.targets.split_targets.values()) == 10_000
    assert sft.targets.train == 9000
    assert sft.targets.validation == 1000
    assert sum(grpo.targets.split_targets.values()) == 1200
    assert grpo.targets.train == 1000
    assert grpo.targets.validation == grpo.targets.test == 100
    assert sft.output.name == "scientific_design_sft_L"
    assert grpo.output.name == "scientific_design_grpo_L"
    assert grpo.input.sft_canonical_path == sft.output.accepted_path
    assert sft.hub.config_name != "scientific_design_sft"
    assert grpo.hub.config_name != "scientific_design_grpo"


def test_generation_push_override_does_not_mutate_original():
    config = load_sft_generation_config(ROOT / "configs" / "generate_sft.toml")
    local = with_generation_push_override(config, push=False)

    assert config.hub.push is True
    assert local.hub.push is False


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


def test_cli_shows_large_generation_config(capsys):
    result = main(
        [
            "show-config",
            "--stage",
            "generate-sft",
            "--config",
            str(ROOT / "configs" / "generate_sft_L.toml"),
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert '"train": 9000' in captured.out
    assert '"validation": 1000' in captured.out
    assert "scientific_design_sft_L" in captured.out
