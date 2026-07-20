from __future__ import annotations

from pathlib import Path
from typing import Any

from trl import SFTConfig, SFTTrainer

from .cli_config import SFTJobConfig, resolve_resume
from .devices import clear_device_cache, detect_runtime
from .hub import require_hf_namespace
from .modeling import (
    load_causal_lm,
    load_tokenizer,
    render_prompt,
    render_sft_completion,
    teaching_lora_config,
)
from .train_common import (
    configure_environment,
    hf_token,
    load_training_dataset,
    require_dataset_schema,
    run_manifest,
    smoke_subset,
    write_json,
    write_jsonl,
)
from .versions import require_training_stack


def run_sft(config: SFTJobConfig, *, smoke_test: bool = False) -> dict[str, Any]:
    configure_environment(config.runtime)
    versions = require_training_stack()
    runtime = detect_runtime()
    token = hf_token()
    output_dir = Path(config.output.directory).expanduser()
    resume_from_checkpoint = resolve_resume(output_dir, config.output.resume)

    if config.hub.push:
        require_hf_namespace(config.hub.repo_id, token=token)

    dataset = load_training_dataset(config.data, token=token)
    require_dataset_schema(
        dataset,
        splits=("train", "validation"),
        columns={"prompt", "completion"},
    )
    if smoke_test:
        dataset = smoke_subset(dataset)

    tokenizer = load_tokenizer(config.model.model_id, token=token)

    def render_row(row: dict[str, Any]) -> dict[str, str]:
        return {
            "prompt": render_prompt(tokenizer, row["prompt"]),
            "completion": render_sft_completion(tokenizer, row["completion"]),
        }

    rendered = dataset.map(
        render_row,
        remove_columns=dataset["train"].column_names,
        desc="Apply the Gemma chat template",
    )
    clear_device_cache(runtime)
    model = load_causal_lm(config.model.model_id, runtime, token=token)
    lora = teaching_lora_config(
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )
    training = config.training
    pad_to_multiple_of = (
        training.mps_pad_to_multiple_of
        if runtime.backend == "mps"
        else training.other_pad_to_multiple_of
    )
    empty_cache_steps = (
        training.mps_empty_cache_steps if runtime.backend == "mps" else None
    )
    pin_memory = (
        runtime.backend == "cuda"
        if config.runtime.pin_memory_on_cuda_only
        else True
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=training.num_train_epochs,
        max_steps=training.max_steps,
        learning_rate=training.learning_rate,
        per_device_train_batch_size=training.train_batch_size,
        per_device_eval_batch_size=training.eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        max_length=training.max_sequence_length,
        pad_to_multiple_of=pad_to_multiple_of,
        completion_only_loss=training.completion_only_loss,
        loss_type=training.loss_type,
        gradient_checkpointing=training.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": training.gradient_checkpointing_use_reentrant
        },
        optim=training.optimizer,
        warmup_steps=training.warmup_steps,
        eval_strategy=training.eval_strategy,
        eval_steps=training.eval_steps,
        save_strategy=training.save_strategy,
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        logging_steps=training.logging_steps,
        logging_first_step=training.logging_first_step,
        report_to=training.report_to,
        torch_empty_cache_steps=empty_cache_steps,
        dataloader_pin_memory=pin_memory,
        push_to_hub=config.hub.push,
        hub_model_id=config.hub.repo_id,
        hub_strategy=config.hub.strategy,
        hub_private_repo=config.hub.private,
        hub_token=token,
        hub_always_push=config.hub.always_push,
        bf16=runtime.trainer_bf16,
        fp16=runtime.trainer_fp16,
        use_cpu=runtime.use_cpu,
        seed=training.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=rendered["train"],
        eval_dataset=rendered["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "resolved_config.json",
        run_manifest(
            stage="sft",
            config=config,
            runtime=runtime,
            versions=versions,
            resume_from_checkpoint=resume_from_checkpoint,
            smoke_test=smoke_test,
        ),
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if config.hub.push:
        trainer.push_to_hub(
            commit_message="Complete scientific design SFT CLI training"
        )
    write_json(output_dir / "train_metrics.json", train_result.metrics)
    write_jsonl(output_dir / "log_history.jsonl", trainer.state.log_history)
    return {
        "output_directory": str(output_dir),
        "hub_repository": config.hub.repo_id if config.hub.push else None,
        "metrics": train_result.metrics,
    }
