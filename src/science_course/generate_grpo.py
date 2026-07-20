from __future__ import annotations

from pathlib import Path
from typing import Any

from .cli_config import GRPOGenerationConfig
from .data import build_grpo_dataset, read_jsonl
from .generate_common import (
    acquire_sources,
    complete_generation_manifest,
    generate_canonical,
    generation_preflight,
    prepare_generation_manifest,
    save_projection,
)


def generate_grpo_dataset(config: GRPOGenerationConfig) -> dict[str, Any]:
    generation_preflight(config)
    sft_canonical_path = Path(config.input.sft_canonical_path).expanduser()
    sft_canonical = read_jsonl(sft_canonical_path)
    if not sft_canonical:
        raise RuntimeError(
            f"No SFT canonical records found at {sft_canonical_path}. "
            "Generate SFT first or choose the correct input path."
        )
    sft_paper_ids = {str(row["paper_id"]) for row in sft_canonical}

    prepare_generation_manifest(config, stage="generate-grpo")
    sources = acquire_sources(config, exclude_paper_ids=sft_paper_ids)
    overlap = {str(row["paper_id"]) for row in sources} & sft_paper_ids
    if overlap:
        raise RuntimeError(
            f"SFT/GRPO source-paper leakage detected for {len(overlap)} papers."
        )
    canonical, _ = generate_canonical(config, sources)
    canonical_overlap = {
        str(row["paper_id"]) for row in canonical
    } & sft_paper_ids
    if canonical_overlap:
        raise RuntimeError(
            f"Accepted GRPO journal overlaps SFT by {len(canonical_overlap)} papers."
        )

    dataset = build_grpo_dataset(canonical)
    forbidden = {"source_text", "completion", "answer", "brainstorm", "principles"}
    for split_name, split_data in dataset.items():
        leaked = forbidden & set(split_data.column_names)
        if leaked:
            raise RuntimeError(
                f"GRPO projection split {split_name!r} leaked columns: "
                + ", ".join(sorted(leaked))
            )

    projection = save_projection(
        dataset,
        config,
        expected_splits={
            "train": config.targets.train,
            "validation": config.targets.validation,
            "test": config.targets.test,
        },
        commit_message=f"Publish {config.output.name} GRPO dataset",
    )
    accepted_journal = read_jsonl(config.output.accepted_path)
    rejected_journal = read_jsonl(config.output.rejected_path)
    result = {
        "name": config.output.name,
        "excluded_sft_papers": len(sft_paper_ids),
        "selected_accepted": len(canonical),
        "accepted_journal": len(accepted_journal),
        "rejected_journal": len(rejected_journal),
        "acceptance_rate": len(accepted_journal)
        / max(len(accepted_journal) + len(rejected_journal), 1),
        **projection,
    }
    complete_generation_manifest(
        config,
        stage="generate-grpo",
        result=result,
    )
    return result
