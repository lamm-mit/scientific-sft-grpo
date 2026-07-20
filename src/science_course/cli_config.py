from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from .data import TASK_FAMILIES


@dataclass(frozen=True)
class RuntimeOptions:
    enable_mps_fallback: bool = True
    tokenizers_parallelism: bool = False
    pin_memory_on_cuda_only: bool = True


@dataclass(frozen=True)
class ModelOptions:
    model_id: str = "google/gemma-4-E4B-it"


@dataclass(frozen=True)
class DatasetOptions:
    source: str = "hub"
    hub_repo: str = "lamm-mit/scientific-sft-grpo-data"
    hub_config: str = ""
    local_path: str = ""

    def __post_init__(self) -> None:
        if self.source not in {"hub", "local"}:
            raise ValueError("data.source must be 'hub' or 'local'.")
        if self.source == "hub" and (not self.hub_repo or not self.hub_config):
            raise ValueError("Hub data requires data.hub_repo and data.hub_config.")
        if self.source == "local" and not self.local_path:
            raise ValueError("Local data requires data.local_path.")


@dataclass(frozen=True)
class OutputOptions:
    directory: str
    resume: str = "auto"

    def __post_init__(self) -> None:
        if not self.directory:
            raise ValueError("output.directory cannot be empty.")
        if not self.resume:
            raise ValueError("output.resume must be 'none', 'auto', or a checkpoint path.")


@dataclass(frozen=True)
class HubOptions:
    repo_id: str
    push: bool = True
    strategy: str = "all_checkpoints"
    private: bool = False
    always_push: bool = True

    def __post_init__(self) -> None:
        if self.push and "/" not in self.repo_id:
            raise ValueError("hub.repo_id must use the form 'namespace/name'.")


@dataclass(frozen=True)
class InferenceOptions:
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass(frozen=True)
class LoraOptions:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1).")


@dataclass(frozen=True)
class SFTTrainingOptions:
    num_train_epochs: float = 3
    max_steps: int = -1
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_sequence_length: int = 1024
    mps_pad_to_multiple_of: int = 128
    other_pad_to_multiple_of: int = 8
    mps_empty_cache_steps: int = 5
    completion_only_loss: bool = True
    loss_type: str = "nll"
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    optimizer: str = "adamw_torch"
    warmup_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 25
    save_strategy: str = "steps"
    save_steps: int = 25
    save_total_limit: int | None = None
    logging_steps: int = 5
    logging_first_step: bool = True
    report_to: str = "none"
    seed: int = 17


@dataclass(frozen=True)
class SFTJobConfig:
    runtime: RuntimeOptions
    model: ModelOptions
    data: DatasetOptions
    output: OutputOptions
    lora: LoraOptions
    training: SFTTrainingOptions
    inference: InferenceOptions
    hub: HubOptions


@dataclass(frozen=True)
class AdapterOptions:
    source: str = "hub"
    hub_repo: str = "lamm-mit/scientific-sft-grpo-design-sft"
    local_path: str = "artifacts/gemma4-scientific-design-sft"

    def __post_init__(self) -> None:
        if self.source not in {"hub", "local"}:
            raise ValueError("adapter.source must be 'hub' or 'local'.")
        if self.source == "hub" and not self.hub_repo:
            raise ValueError("Hub adapter loading requires adapter.hub_repo.")
        if self.source == "local" and not self.local_path:
            raise ValueError("Local adapter loading requires adapter.local_path.")

    @property
    def selected_source(self) -> str:
        return self.hub_repo if self.source == "hub" else self.local_path


@dataclass(frozen=True)
class RewardOptions:
    judge_model: str = "gpt-5.6-luna"
    cache_path: str = "results/scientific_design_judge_cache.jsonl"
    format_base_reward: float = 0.10
    semantic_reward_weight: float = 0.90
    function_weights: tuple[float, ...] = (1.0,)
    openai_max_retries: int = 3
    openai_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if abs(self.format_base_reward + self.semantic_reward_weight - 1.0) > 1e-9:
            raise ValueError("Reward coefficients must sum to one.")
        if self.format_base_reward < 0 or self.semantic_reward_weight < 0:
            raise ValueError("Reward coefficients cannot be negative.")
        if not self.function_weights:
            raise ValueError("reward.function_weights cannot be empty.")


@dataclass(frozen=True)
class GRPOTrainingOptions:
    num_train_epochs: float = 1
    max_steps: int = -1
    learning_rate: float = 5e-6
    train_batch_size: int = 1
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    num_generations_eval: int = 4
    max_completion_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    beta: float = 0.0
    scale_rewards: str = "group"
    multi_objective_aggregation: str = "sum_then_normalize"
    loss_type: str = "dapo"
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    optimizer: str = "adamw_torch"
    warmup_steps: int = 5
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    save_strategy: str = "steps"
    save_steps: int = 25
    save_total_limit: int | None = None
    logging_steps: int = 1
    logging_first_step: bool = True
    log_completions: bool = True
    num_completions_to_print: int = 4
    report_to: str = "none"
    use_vllm: bool = False
    seed: int = 17

    def __post_init__(self) -> None:
        if self.num_generations <= 1 or self.num_generations_eval <= 1:
            raise ValueError("GRPO requires at least two generations per prompt.")
        if self.eval_batch_size % self.num_generations_eval:
            raise ValueError(
                "training.eval_batch_size must be divisible by "
                "training.num_generations_eval."
            )


@dataclass(frozen=True)
class GRPOJobConfig:
    runtime: RuntimeOptions
    model: ModelOptions
    data: DatasetOptions
    adapter: AdapterOptions
    output: OutputOptions
    reward: RewardOptions
    training: GRPOTrainingOptions
    inference: InferenceOptions
    hub: HubOptions


@dataclass(frozen=True)
class GenerationModelsOptions:
    teacher_model: str = "gpt-5.6-terra"
    critic_model: str = "gpt-5.6-terra"


@dataclass(frozen=True)
class SourcePoolOptions:
    dataset_id: str = "common-pile/peS2o"
    split: str = "train"
    candidate_limit: int = 2500
    max_records_scanned: int = 200_000
    min_chars: int = 1200
    max_chars: int = 6000
    seed: int = 17

    def __post_init__(self) -> None:
        if self.candidate_limit <= 0 or self.max_records_scanned <= 0:
            raise ValueError("Source candidate and scan limits must be positive.")
        if not 0 < self.min_chars < self.max_chars:
            raise ValueError("Source character limits must satisfy 0 < min_chars < max_chars.")


@dataclass(frozen=True)
class GenerationAPIOptions:
    concurrency: int = 8
    max_attempts: int = 2500
    openai_max_retries: int = 3
    openai_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("generation.concurrency must be at least one.")
        if self.max_attempts < 1:
            raise ValueError("generation.max_attempts must be positive.")
        if self.openai_max_retries < 0 or self.openai_timeout_seconds <= 0:
            raise ValueError("OpenAI retry and timeout settings are invalid.")


@dataclass(frozen=True)
class GenerationOutputOptions:
    name: str
    raw_sources_path: str
    accepted_path: str
    rejected_path: str
    dataset_directory: str
    parquet_directory: str = "data/processed"
    parquet_prefix: str = ""
    manifest_path: str = ""

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError("output.name must be a non-empty filesystem-safe name.")
        required = (
            self.raw_sources_path,
            self.accepted_path,
            self.rejected_path,
            self.dataset_directory,
            self.parquet_prefix,
            self.manifest_path,
        )
        if not all(required):
            raise ValueError("All generation output paths and prefixes must be configured.")


@dataclass(frozen=True)
class DatasetPublicationOptions:
    repo_id: str = "lamm-mit/scientific-sft-grpo-data"
    config_name: str = ""
    push: bool = True
    private: bool = False

    def __post_init__(self) -> None:
        if self.push and "/" not in self.repo_id:
            raise ValueError("hub.repo_id must use the form 'namespace/name'.")
        if not self.config_name:
            raise ValueError("hub.config_name cannot be empty.")


@dataclass(frozen=True)
class SFTGenerationTargets:
    train: int = 500
    validation: int = 75

    def __post_init__(self) -> None:
        if self.train <= 0 or self.validation <= 0:
            raise ValueError("SFT train and validation targets must be positive.")

    @property
    def split_targets(self) -> dict[str, int]:
        return {"sft_train": self.train, "sft_validation": self.validation}


@dataclass(frozen=True)
class GRPOGenerationTargets:
    train: int = 500
    validation: int = 75
    test: int = 100

    def __post_init__(self) -> None:
        if self.train <= 0 or self.validation <= 0 or self.test <= 0:
            raise ValueError("GRPO train, validation, and test targets must be positive.")

    @property
    def split_targets(self) -> dict[str, int]:
        return {
            "grpo_train": self.train,
            "grpo_validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class GRPOGenerationInputOptions:
    sft_canonical_path: str = "data/canonical/scientific_design_sft_tasks.jsonl"

    def __post_init__(self) -> None:
        if not self.sft_canonical_path:
            raise ValueError("input.sft_canonical_path cannot be empty.")


@dataclass(frozen=True)
class SFTGenerationConfig:
    models: GenerationModelsOptions
    targets: SFTGenerationTargets
    task_family_weights: dict[str, float]
    source: SourcePoolOptions
    generation: GenerationAPIOptions
    output: GenerationOutputOptions
    hub: DatasetPublicationOptions


@dataclass(frozen=True)
class GRPOGenerationConfig:
    models: GenerationModelsOptions
    targets: GRPOGenerationTargets
    task_family_weights: dict[str, float]
    source: SourcePoolOptions
    generation: GenerationAPIOptions
    input: GRPOGenerationInputOptions
    output: GenerationOutputOptions
    hub: DatasetPublicationOptions


ConfigType = TypeVar("ConfigType")


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table.")
    return value


def _construct(
    cls: type[ConfigType],
    values: dict[str, Any],
    section: str,
) -> ConfigType:
    field_names = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(values) - field_names)
    if unknown:
        raise ValueError(f"Unknown keys in [{section}]: {', '.join(unknown)}")
    try:
        return cls(**values)
    except TypeError as error:
        raise ValueError(f"Invalid [{section}] configuration: {error}") from error


def _document(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    with source.open("rb") as handle:
        document = tomllib.load(handle)
    if not isinstance(document, dict):
        raise ValueError("The TOML root must be a table.")
    return document


def _validate_task_family_weights(values: dict[str, Any]) -> dict[str, float]:
    if set(values) != set(TASK_FAMILIES):
        raise ValueError(
            "The [task_family_weights] table must define exactly: "
            + ", ".join(TASK_FAMILIES)
        )
    weights = {name: float(value) for name, value in values.items()}
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Task-family weights must be non-negative with a positive sum.")
    return weights


def _validate_generation_scale(
    *,
    target_total: int,
    source: SourcePoolOptions,
    generation: GenerationAPIOptions,
) -> None:
    if source.candidate_limit < target_total:
        raise ValueError(
            "source.candidate_limit cannot be smaller than the accepted-example total."
        )
    if generation.max_attempts < target_total:
        raise ValueError(
            "generation.max_attempts cannot be smaller than the accepted-example total."
        )


def load_sft_config(path: str | Path) -> SFTJobConfig:
    document = _document(path)
    return SFTJobConfig(
        runtime=_construct(RuntimeOptions, _table(document, "runtime"), "runtime"),
        model=_construct(ModelOptions, _table(document, "model"), "model"),
        data=_construct(DatasetOptions, _table(document, "data"), "data"),
        output=_construct(OutputOptions, _table(document, "output"), "output"),
        lora=_construct(LoraOptions, _table(document, "lora"), "lora"),
        training=_construct(
            SFTTrainingOptions,
            _table(document, "training"),
            "training",
        ),
        inference=_construct(
            InferenceOptions,
            _table(document, "inference"),
            "inference",
        ),
        hub=_construct(HubOptions, _table(document, "hub"), "hub"),
    )


def load_grpo_config(path: str | Path) -> GRPOJobConfig:
    document = _document(path)
    reward_values = dict(_table(document, "reward"))
    if "function_weights" in reward_values:
        reward_values["function_weights"] = tuple(reward_values["function_weights"])
    return GRPOJobConfig(
        runtime=_construct(RuntimeOptions, _table(document, "runtime"), "runtime"),
        model=_construct(ModelOptions, _table(document, "model"), "model"),
        data=_construct(DatasetOptions, _table(document, "data"), "data"),
        adapter=_construct(AdapterOptions, _table(document, "adapter"), "adapter"),
        output=_construct(OutputOptions, _table(document, "output"), "output"),
        reward=_construct(RewardOptions, reward_values, "reward"),
        training=_construct(
            GRPOTrainingOptions,
            _table(document, "training"),
            "training",
        ),
        inference=_construct(
            InferenceOptions,
            _table(document, "inference"),
            "inference",
        ),
        hub=_construct(HubOptions, _table(document, "hub"), "hub"),
    )


def load_sft_generation_config(path: str | Path) -> SFTGenerationConfig:
    document = _document(path)
    targets = _construct(
        SFTGenerationTargets,
        _table(document, "targets"),
        "targets",
    )
    source = _construct(SourcePoolOptions, _table(document, "source"), "source")
    generation = _construct(
        GenerationAPIOptions,
        _table(document, "generation"),
        "generation",
    )
    _validate_generation_scale(
        target_total=sum(targets.split_targets.values()),
        source=source,
        generation=generation,
    )
    return SFTGenerationConfig(
        models=_construct(
            GenerationModelsOptions,
            _table(document, "models"),
            "models",
        ),
        targets=targets,
        task_family_weights=_validate_task_family_weights(
            _table(document, "task_family_weights")
        ),
        source=source,
        generation=generation,
        output=_construct(
            GenerationOutputOptions,
            _table(document, "output"),
            "output",
        ),
        hub=_construct(
            DatasetPublicationOptions,
            _table(document, "hub"),
            "hub",
        ),
    )


def load_grpo_generation_config(path: str | Path) -> GRPOGenerationConfig:
    document = _document(path)
    targets = _construct(
        GRPOGenerationTargets,
        _table(document, "targets"),
        "targets",
    )
    source = _construct(SourcePoolOptions, _table(document, "source"), "source")
    generation = _construct(
        GenerationAPIOptions,
        _table(document, "generation"),
        "generation",
    )
    _validate_generation_scale(
        target_total=sum(targets.split_targets.values()),
        source=source,
        generation=generation,
    )
    return GRPOGenerationConfig(
        models=_construct(
            GenerationModelsOptions,
            _table(document, "models"),
            "models",
        ),
        targets=targets,
        task_family_weights=_validate_task_family_weights(
            _table(document, "task_family_weights")
        ),
        source=source,
        generation=generation,
        input=_construct(
            GRPOGenerationInputOptions,
            _table(document, "input"),
            "input",
        ),
        output=_construct(
            GenerationOutputOptions,
            _table(document, "output"),
            "output",
        ),
        hub=_construct(
            DatasetPublicationOptions,
            _table(document, "hub"),
            "hub",
        ),
    )


AnyJobConfig = (
    SFTJobConfig
    | GRPOJobConfig
    | SFTGenerationConfig
    | GRPOGenerationConfig
)


def config_as_dict(config: AnyJobConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)


def with_cli_overrides(
    config: SFTJobConfig | GRPOJobConfig,
    *,
    resume: str | None,
    push: bool | None,
    smoke_test: bool,
) -> SFTJobConfig | GRPOJobConfig:
    output = config.output
    hub = config.hub
    training = config.training
    if resume is not None:
        output = replace(output, resume=resume)
    if push is not None:
        hub = replace(hub, push=push)
    if smoke_test:
        output = replace(
            output,
            directory=str(Path(str(output.directory) + "-smoke")),
            resume="none",
        )
        hub = replace(hub, push=False)
        training = replace(training, max_steps=1)
    return replace(config, output=output, hub=hub, training=training)


def with_generation_push_override(
    config: SFTGenerationConfig | GRPOGenerationConfig,
    *,
    push: bool | None,
) -> SFTGenerationConfig | GRPOGenerationConfig:
    return config if push is None else replace(config, hub=replace(config.hub, push=push))


def latest_checkpoint(output_directory: str | Path) -> Path | None:
    output = Path(output_directory)
    candidates: list[tuple[int, Path]] = []
    for path in output.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def resolve_resume(
    output_directory: str | Path,
    resume: str,
) -> str | None:
    normalized = resume.strip().lower()
    if normalized in {"none", "false", "no"}:
        return None
    if normalized == "auto":
        checkpoint = latest_checkpoint(output_directory)
        return str(checkpoint) if checkpoint is not None else None
    checkpoint = Path(resume).expanduser()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    return str(checkpoint)
