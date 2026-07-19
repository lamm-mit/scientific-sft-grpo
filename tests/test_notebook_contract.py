from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def _source(name: str) -> str:
    artifact = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    return "\n".join(
        cell.source for cell in artifact.cells if cell.cell_type == "code"
    )


def test_dataset_notebooks_expose_scale_and_api_parameters():
    sft = _source("01_generate_scientific_design_sft_dataset.ipynb")
    grpo = _source("02_generate_scientific_design_grpo_dataset.ipynb")

    for source in (sft, grpo):
        assert "TASK_FAMILY_WEIGHTS =" in source
        assert "SOURCE_CANDIDATE_LIMIT =" in source
        assert "GENERATION_CONCURRENCY =" in source
        assert "MAX_GENERATION_ATTEMPTS =" in source
        assert "OPENAI_MAX_RETRIES =" in source
        assert "OPENAI_TIMEOUT_SECONDS =" in source
        assert "DATASET_HF_REPO =" in source
        assert 'HF_TOKEN = None' in source
    assert '"sft_train": 500' in sft
    assert '"grpo_train": 500' in grpo
    assert '"test": 100' in grpo
    assert 'JUDGE_MODEL = DEFAULT_JUDGE_MODEL' in grpo


def test_training_notebooks_expose_models_training_and_hub_parameters():
    sft = _source("03_finetune_sft_lora.ipynb")
    grpo = _source("04_finetune_grpo_lora.ipynb")

    for source in (sft, grpo):
        assert 'MODEL_ID = "google/gemma-4-E4B-it"' in source
        assert "MAX_STEPS =" in source
        assert "LEARNING_RATE =" in source
        assert "GRADIENT_ACCUMULATION_STEPS =" in source
        assert "EVAL_STEPS =" in source
        assert "SAVE_STEPS =" in source
        assert "RESUME_FROM_CHECKPOINT = None" in source
        assert "MAX_NEW_TOKENS =" in source
        assert "PUSH_TO_HUB = True" in source
        assert "HUB_STRATEGY =" in source
        assert 'HF_TOKEN = None' in source
    assert "LORA_RANK =" in sft
    assert 'SFT_DATA_SOURCE_MODE = "hub"' in sft
    assert "load_dataset(" in sft
    assert 'JUDGE_MODEL = DEFAULT_JUDGE_MODEL' in grpo
    assert 'GRPO_DATA_SOURCE_MODE = "hub"' in grpo
    assert 'SFT_ADAPTER_SOURCE_MODE = "hub"' in grpo
    assert "load_dataset(" in grpo
    assert "SFT_ADAPTER_SOURCE = (" in grpo
    assert "FORMAT_BASE_REWARD = 0.10" in grpo
    assert "SEMANTIC_REWARD_WEIGHT = 0.90" in grpo
    assert "NUM_GENERATIONS = 4" in grpo
