from __future__ import annotations

from typing import Any

from .cli_config import SFTGenerationConfig
from .data import build_sft_dataset, read_jsonl
from .generate_common import (
    acquire_sources,
    complete_generation_manifest,
    generate_canonical,
    generation_preflight,
    prepare_generation_manifest,
    save_projection,
)


def generate_sft_dataset(config: SFTGenerationConfig) -> dict[str, Any]:
    generation_preflight(config)
    prepare_generation_manifest(config, stage="generate-sft")
    sources = acquire_sources(config)
    canonical, _ = generate_canonical(config, sources)
    dataset = build_sft_dataset(canonical)
    if "source_text" in dataset["train"].column_names:
        raise RuntimeError("SFT projection leaked source_text.")
    if "required_constraints" in dataset["train"].column_names:
        raise RuntimeError("SFT projection leaked the hidden grading rubric.")

    projection = save_projection(
        dataset,
        config,
        expected_splits={
            "train": config.targets.train,
            "validation": config.targets.validation,
        },
        commit_message=f"Publish {config.output.name} SFT dataset",
    )
    accepted_journal = read_jsonl(config.output.accepted_path)
    rejected_journal = read_jsonl(config.output.rejected_path)
    result = {
        "name": config.output.name,
        "selected_accepted": len(canonical),
        "accepted_journal": len(accepted_journal),
        "rejected_journal": len(rejected_journal),
        "acceptance_rate": len(accepted_journal)
        / max(len(accepted_journal) + len(rejected_journal), 1),
        **projection,
    }
    complete_generation_manifest(
        config,
        stage="generate-sft",
        result=result,
    )
    return result
