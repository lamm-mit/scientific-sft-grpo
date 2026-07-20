from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def notebook(cells: list):
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (scientific-sft-grpo)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
    )


nb1 = notebook(
    [
        markdown(
            """
            # 01 — Generate the scientific problem-solving SFT dataset

            The student model receives a **task only**—never a paper passage. Openly licensed
            scientific text is authoring material for `gpt-5.6-terra`, which creates a new,
            self-contained task and one strong structured work product:

            ```text
            <brainstorm>distinct candidate ideas</brainstorm>
            <principles>scientific constraints and design principles</principles>
            <synthesis>comparison or integration of the candidates</synthesis>
            <answer>concise final proposal or conclusion</answer>
            ```

            A second critic call checks every generated record against the source. The
            generator continues until the requested number of **accepted** examples is
            available in every task family and SFT split.
            """
        ),
        markdown(
            """
            ## Before you run

            ```bash
            python3.12 -m venv .venv
            source .venv/bin/activate
            python -m pip install --upgrade pip
            python -m pip install -e ".[dev]"
            export OPENAI_API_KEY="..."
            hf auth login
            jupyter lab
            ```

            Generation is real API use. Drafts and rejections are appended incrementally, so
            interruption and rerun are safe. Increasing the targets may require increasing
            the source pool and generation-attempt budget.
            """
        ),
        markdown(
            """
            ## Configuration

            Every editable non-secret parameter is below. Only `OPENAI_API_KEY` is required
            from the environment. Hugging Face uses cached `hf auth login` credentials unless
            the optional token line is deliberately uncommented.
            """
        ),
        code(
            """
            import shutil
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            from IPython.display import JSON, Markdown, display

            from science_course.data import (
                TASK_FAMILIES,
                allocate_task_quotas,
                build_sft_dataset,
                read_jsonl,
                stream_open_science_sources,
                task_quota_status,
                write_jsonl,
            )
            from science_course.hub import require_hf_namespace
            from science_course.teacher import (
                DEFAULT_CRITIC_MODEL,
                DEFAULT_TEACHER_MODEL,
                generate_canonical_tasks,
                require_openai_key,
            )

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            DATA = ROOT / "data"

            # Local artifacts: v2 paths prevent accidental reuse of the earlier QA dataset
            RAW_SOURCES = DATA / "raw" / "scientific_design_sft_sources.jsonl"
            ACCEPTED = DATA / "canonical" / "scientific_design_sft_tasks.jsonl"
            REJECTED = DATA / "canonical" / "scientific_design_sft_rejected.jsonl"
            SFT_DISK = DATA / "processed" / "scientific_design_sft"

            # Teacher and independent critic
            TEACHER_MODEL = DEFAULT_TEACHER_MODEL
            CRITIC_MODEL = DEFAULT_CRITIC_MODEL

            # Exact accepted-example targets
            SFT_SPLIT_TARGETS = {
                "sft_train": 500,
                "sft_validation": 75,
            }
            TASK_FAMILY_WEIGHTS = {
                "mechanism_guided_design": 0.25,
                "experimental_design": 0.25,
                "troubleshooting": 0.20,
                "hypothesis_development": 0.15,
                "cross_domain_synthesis": 0.15,
            }

            # Open scientific source pool
            SOURCE_DATASET_ID = "common-pile/peS2o"
            SOURCE_SPLIT = "train"
            SOURCE_CANDIDATE_LIMIT = 2_500
            MAX_SOURCE_RECORDS_SCANNED = 200_000
            MIN_SOURCE_CHARS = 1_200
            MAX_SOURCE_CHARS = 6_000
            RANDOM_SEED = 17

            # Resumable API generation
            GENERATION_CONCURRENCY = 8
            MAX_GENERATION_ATTEMPTS = 2_500
            OPENAI_MAX_RETRIES = 3
            OPENAI_TIMEOUT_SECONDS = 120.0

            # Hugging Face publication
            DATASET_HF_REPO = "lamm-mit/scientific-sft-grpo-data"
            DATASET_CONFIG_NAME = "scientific_design_sft"
            PUSH_DATASETS_TO_HUB = True
            DATASET_PRIVATE = False
            HF_TOKEN = None
            # HF_TOKEN = __import__("os").environ["HF_TOKEN"]  # Optional; prefer `hf auth login`.

            require_openai_key()
            if set(TASK_FAMILY_WEIGHTS) != set(TASK_FAMILIES):
                raise ValueError(f"Define weights for exactly these families: {TASK_FAMILIES}")
            if PUSH_DATASETS_TO_HUB:
                require_hf_namespace(DATASET_HF_REPO, token=HF_TOKEN)

            sns.set_theme(style="whitegrid", context="talk")
            display(
                JSON(
                    {
                        "teacher_model": TEACHER_MODEL,
                        "critic_model": CRITIC_MODEL,
                        "accepted_targets": SFT_SPLIT_TARGETS,
                        "accepted_total": sum(SFT_SPLIT_TARGETS.values()),
                        "source_candidates": SOURCE_CANDIDATE_LIMIT,
                        "generation_concurrency": GENERATION_CONCURRENCY,
                        "dataset_hub_repo": DATASET_HF_REPO,
                        "dataset_config": DATASET_CONFIG_NAME,
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 1. Acquire open scientific authoring material

            Only license-filtered source excerpts are sent to the teacher. Paper ID, URL,
            license, and content hash are retained locally for provenance. The source text is
            never placed in the student prompt or published training projection.
            """
        ),
        code(
            """
            cached_sources = read_jsonl(RAW_SOURCES)
            cache_is_usable = (
                len(cached_sources) == SOURCE_CANDIDATE_LIMIT
                and all(
                    row.get("source_dataset") == SOURCE_DATASET_ID
                    and row.get("source_split") == SOURCE_SPLIT
                    for row in cached_sources
                )
            )
            if cache_is_usable:
                sources = cached_sources
                print(f"Reusing {len(sources)} cached sources from {RAW_SOURCES}")
            else:
                sources = stream_open_science_sources(
                    dataset_id=SOURCE_DATASET_ID,
                    split=SOURCE_SPLIT,
                    max_papers=SOURCE_CANDIDATE_LIMIT,
                    max_scanned=MAX_SOURCE_RECORDS_SCANNED,
                    min_chars=MIN_SOURCE_CHARS,
                    max_chars=MAX_SOURCE_CHARS,
                    seed=RANDOM_SEED,
                )
                write_jsonl(RAW_SOURCES, sources)
                print(f"Cached {len(sources)} source candidates at {RAW_SOURCES}")

            if len(sources) < sum(SFT_SPLIT_TARGETS.values()):
                raise RuntimeError(
                    "The source pool is smaller than the accepted-example target. "
                    "Increase SOURCE_CANDIDATE_LIMIT."
                )
            source_frame = pd.DataFrame(sources)
            display(
                source_frame[
                    ["paper_id", "title", "source_license", "source_url"]
                ].head()
            )
            """
        ),
        code(
            """
            quota = allocate_task_quotas(SFT_SPLIT_TARGETS, TASK_FAMILY_WEIGHTS)
            quota_frame = pd.DataFrame(
                [
                    {"split": split, "task_family": family, "target": target}
                    for (split, family), target in quota.items()
                ]
            )
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            source_frame["source_license"].value_counts().head(12).plot.bar(
                ax=axes[0], color="#315c8c", title="Open-license provenance"
            )
            sns.barplot(
                quota_frame,
                x="task_family",
                y="target",
                hue="split",
                ax=axes[1],
            )
            axes[1].set_title("Accepted-example quotas")
            axes[1].tick_params(axis="x", rotation=35)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 2. Generate until the accepted quotas are full

            Each source is assigned to the currently most underfilled `(split, task_family)`
            cell. Rejected drafts do not consume a quota. Concurrent calls improve throughput,
            while all completed records are written immediately for safe resumption.
            """
        ),
        code(
            """
            canonical = generate_canonical_tasks(
                sources,
                accepted_path=ACCEPTED,
                rejected_path=REJECTED,
                split_targets=SFT_SPLIT_TARGETS,
                task_family_weights=TASK_FAMILY_WEIGHTS,
                teacher_model=TEACHER_MODEL,
                critic_model=CRITIC_MODEL,
                concurrency=GENERATION_CONCURRENCY,
                max_attempts=MAX_GENERATION_ATTEMPTS,
                api_max_retries=OPENAI_MAX_RETRIES,
                api_timeout_seconds=OPENAI_TIMEOUT_SECONDS,
            )
            rejected = read_jsonl(REJECTED)
            status = task_quota_status(
                canonical,
                SFT_SPLIT_TARGETS,
                TASK_FAMILY_WEIGHTS,
            )
            if not status["complete"]:
                raise RuntimeError(f"Unfilled quotas: {status['deficits']}")
            print(
                {
                    "accepted": len(canonical),
                    "rejected": len(rejected),
                    "acceptance_rate": len(canonical)
                    / max(len(canonical) + len(rejected), 1),
                    "split_counts": pd.Series(
                        [row["split"] for row in canonical]
                    ).value_counts().to_dict(),
                }
            )
            """
        ),
        code(
            """
            example = canonical[0]
            display(Markdown("### Student-visible task"))
            display(Markdown(example["task"]))
            display(Markdown("### Reference structured work product"))
            display(
                JSON(
                    {
                        "brainstorm": example["brainstorm"],
                        "principles": example["principles"],
                        "synthesis": example["synthesis"],
                        "answer": example["answer"],
                    }
                )
            )
            display(Markdown("### Hidden task-specific rubric"))
            display(
                JSON(
                    {
                        "required_constraints": example["required_constraints"],
                        "evaluation_criteria": example["evaluation_criteria"],
                        "acceptable_alternatives": example["acceptable_alternatives"],
                        "failure_modes": example["failure_modes"],
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 3. Audit the curriculum

            The audit checks family balance and whether the reference work products contain
            multiple candidates and explicit principles. It does not assume that longer
            answers are better.
            """
        ),
        code(
            """
            audit = pd.DataFrame(
                [
                    {
                        "split": row["split"],
                        "task_family": row["task_family"],
                        "task_chars": len(row["task"]),
                        "brainstorm_ideas": len(row["brainstorm"]),
                        "principles": len(row["principles"]),
                    }
                    for row in canonical
                ]
            )
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            sns.countplot(
                audit,
                x="task_family",
                hue="split",
                ax=axes[0],
            )
            axes[0].tick_params(axis="x", rotation=35)
            axes[0].set_title("Accepted task-family balance")
            sns.scatterplot(
                audit,
                x="brainstorm_ideas",
                y="principles",
                hue="task_family",
                alpha=0.75,
                ax=axes[1],
            )
            axes[1].set_title("Structured reference depth")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 4. Build and publish the SFT projection

            SFT receives `prompt + completion`. Raw source text, critic metadata, and the
            hidden grading rubric are excluded from this projection.
            """
        ),
        code(
            """
            sft = build_sft_dataset(canonical)
            expected_splits = {
                "train": SFT_SPLIT_TARGETS["sft_train"],
                "validation": SFT_SPLIT_TARGETS["sft_validation"],
            }
            actual_splits = {name: len(split) for name, split in sft.items()}
            if actual_splits != expected_splits:
                raise RuntimeError(
                    f"Unexpected SFT split sizes: {actual_splits}; expected {expected_splits}"
                )
            if SFT_DISK.exists():
                shutil.rmtree(SFT_DISK)
            sft.save_to_disk(SFT_DISK)
            for split_name, split_data in sft.items():
                split_data.to_parquet(
                    DATA / "processed" / f"scientific_design_sft_{split_name}.parquet"
                )
            if PUSH_DATASETS_TO_HUB:
                sft.push_to_hub(
                    DATASET_HF_REPO,
                    config_name=DATASET_CONFIG_NAME,
                    private=DATASET_PRIVATE,
                    token=HF_TOKEN,
                    commit_message="Publish scientific design SFT dataset",
                )

            assert "source_text" not in sft["train"].column_names
            assert "required_constraints" not in sft["train"].column_names
            print(sft)
            display(JSON(sft["train"][0]))
            """
        ),
        markdown(
            """
            ## Result

            The SFT dataset contains 500 training and 75 validation tasks by default, balanced
            across five scientific problem-solving families. Continue with notebook 02, which
            generates a separate, paper-disjoint GRPO curriculum and its hidden rubrics.
            """
        ),
    ]
)


nb2 = notebook(
    [
        markdown(
            """
            # 02 — Generate the scientific problem-solving GRPO dataset

            This notebook builds a separate task bank for GRPO. It excludes every paper used
            for SFT, generates task-only prompts plus hidden task-specific rubrics, and
            demonstrates the exact reward:

            \\[
            R = F\\times(0.10 + 0.90J)
            \\]

            `F` is an exact four-section format gate. `J` is the average of four integer
            0–4 scores from `gpt-5.6-luna`: brainstorm, principles, synthesis, and answer.
            """
        ),
        markdown(
            """
            ## Configuration

            Dataset-generation and reward parameters are explicit below. `OPENAI_API_KEY`
            remains an environment secret; Hugging Face uses cached login unless the optional
            token line is uncommented.
            """
        ),
        code(
            """
            import shutil
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            from IPython.display import JSON, Markdown, display

            from science_course.data import (
                TASK_FAMILIES,
                build_grpo_dataset,
                read_jsonl,
                render_completion,
                stream_open_science_sources,
                task_quota_status,
                write_jsonl,
            )
            from science_course.hub import require_hf_namespace
            from science_course.judge import (
                DEFAULT_JUDGE_MODEL,
                ScientificDesignJudge,
            )
            from science_course.rewards import (
                combined_reward,
                format_reward,
                semantic_score_from_judgment,
            )
            from science_course.teacher import (
                DEFAULT_CRITIC_MODEL,
                DEFAULT_TEACHER_MODEL,
                generate_canonical_tasks,
                require_openai_key,
            )

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            DATA = ROOT / "data"

            # Local artifacts
            SFT_CANONICAL = DATA / "canonical" / "scientific_design_sft_tasks.jsonl"
            RAW_SOURCES = DATA / "raw" / "scientific_design_grpo_sources.jsonl"
            ACCEPTED = DATA / "canonical" / "scientific_design_grpo_tasks.jsonl"
            REJECTED = DATA / "canonical" / "scientific_design_grpo_rejected.jsonl"
            GRPO_DISK = DATA / "processed" / "scientific_design_grpo"
            JUDGE_CACHE = ROOT / "results" / "scientific_design_judge_cache.jsonl"

            # Teacher, critic, and GRPO judge
            TEACHER_MODEL = DEFAULT_TEACHER_MODEL
            CRITIC_MODEL = DEFAULT_CRITIC_MODEL
            JUDGE_MODEL = DEFAULT_JUDGE_MODEL

            # Exact accepted-example targets
            GRPO_SPLIT_TARGETS = {
                "grpo_train": 500,
                "grpo_validation": 75,
                "test": 100,
            }
            TASK_FAMILY_WEIGHTS = {
                "mechanism_guided_design": 0.25,
                "experimental_design": 0.25,
                "troubleshooting": 0.20,
                "hypothesis_development": 0.15,
                "cross_domain_synthesis": 0.15,
            }

            # Open scientific source pool
            SOURCE_DATASET_ID = "common-pile/peS2o"
            SOURCE_SPLIT = "train"
            SOURCE_CANDIDATE_LIMIT = 3_000
            MAX_SOURCE_RECORDS_SCANNED = 250_000
            MIN_SOURCE_CHARS = 1_200
            MAX_SOURCE_CHARS = 6_000
            RANDOM_SEED = 29

            # Resumable API generation
            GENERATION_CONCURRENCY = 8
            MAX_GENERATION_ATTEMPTS = 3_000
            OPENAI_MAX_RETRIES = 3
            OPENAI_TIMEOUT_SECONDS = 120.0

            # Exact GRPO reward coefficients
            FORMAT_BASE_REWARD = 0.10
            SEMANTIC_REWARD_WEIGHT = 0.90
            JUDGE_DIMENSION_MAX_SCORE = 4

            # Hugging Face publication
            DATASET_HF_REPO = "lamm-mit/scientific-sft-grpo-data"
            DATASET_CONFIG_NAME = "scientific_design_grpo"
            PUSH_DATASETS_TO_HUB = True
            DATASET_PRIVATE = False
            HF_TOKEN = None
            # HF_TOKEN = __import__("os").environ["HF_TOKEN"]  # Optional; prefer `hf auth login`.

            require_openai_key()
            if set(TASK_FAMILY_WEIGHTS) != set(TASK_FAMILIES):
                raise ValueError(f"Define weights for exactly these families: {TASK_FAMILIES}")
            if abs(FORMAT_BASE_REWARD + SEMANTIC_REWARD_WEIGHT - 1.0) > 1e-9:
                raise ValueError("The reward coefficients must sum to one.")
            if JUDGE_DIMENSION_MAX_SCORE != 4:
                raise ValueError("The tested Luna rubric uses integer scores from 0 to 4.")
            if PUSH_DATASETS_TO_HUB:
                require_hf_namespace(DATASET_HF_REPO, token=HF_TOKEN)

            sft_canonical = read_jsonl(SFT_CANONICAL)
            if not sft_canonical:
                raise RuntimeError("Run notebook 01 first.")
            sft_paper_ids = {row["paper_id"] for row in sft_canonical}
            sns.set_theme(style="whitegrid", context="talk")
            display(
                JSON(
                    {
                        "teacher_model": TEACHER_MODEL,
                        "critic_model": CRITIC_MODEL,
                        "judge_model": JUDGE_MODEL,
                        "accepted_targets": GRPO_SPLIT_TARGETS,
                        "excluded_sft_papers": len(sft_paper_ids),
                        "reward_formula": "F * (0.10 + 0.90 * J)",
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 1. Acquire a paper-disjoint source pool

            Every paper used by SFT is excluded before GRPO task generation. This makes the
            two adaptation stages and final test set traceable to disjoint source documents.
            """
        ),
        code(
            """
            cached_sources = read_jsonl(RAW_SOURCES)
            cache_is_usable = (
                len(cached_sources) == SOURCE_CANDIDATE_LIMIT
                and not ({row["paper_id"] for row in cached_sources} & sft_paper_ids)
                and all(
                    row.get("source_dataset") == SOURCE_DATASET_ID
                    and row.get("source_split") == SOURCE_SPLIT
                    for row in cached_sources
                )
            )
            if cache_is_usable:
                sources = cached_sources
                print(f"Reusing {len(sources)} cached GRPO sources")
            else:
                sources = stream_open_science_sources(
                    dataset_id=SOURCE_DATASET_ID,
                    split=SOURCE_SPLIT,
                    max_papers=SOURCE_CANDIDATE_LIMIT,
                    max_scanned=MAX_SOURCE_RECORDS_SCANNED,
                    min_chars=MIN_SOURCE_CHARS,
                    max_chars=MAX_SOURCE_CHARS,
                    seed=RANDOM_SEED,
                    exclude_paper_ids=sft_paper_ids,
                )
                write_jsonl(RAW_SOURCES, sources)
                print(f"Cached {len(sources)} paper-disjoint source candidates")
            if {row["paper_id"] for row in sources} & sft_paper_ids:
                raise RuntimeError("SFT/GRPO source-paper leakage detected.")
            """
        ),
        markdown(
            """
            ## 2. Generate GRPO tasks and hidden rubrics

            GRPO references are retained only in the local canonical audit. The policy-facing
            projection contains the task and a hidden rubric for reward calculation—never a
            gold completion.
            """
        ),
        code(
            """
            canonical = generate_canonical_tasks(
                sources,
                accepted_path=ACCEPTED,
                rejected_path=REJECTED,
                split_targets=GRPO_SPLIT_TARGETS,
                task_family_weights=TASK_FAMILY_WEIGHTS,
                teacher_model=TEACHER_MODEL,
                critic_model=CRITIC_MODEL,
                concurrency=GENERATION_CONCURRENCY,
                max_attempts=MAX_GENERATION_ATTEMPTS,
                api_max_retries=OPENAI_MAX_RETRIES,
                api_timeout_seconds=OPENAI_TIMEOUT_SECONDS,
            )
            status = task_quota_status(
                canonical,
                GRPO_SPLIT_TARGETS,
                TASK_FAMILY_WEIGHTS,
            )
            if not status["complete"]:
                raise RuntimeError(f"Unfilled GRPO quotas: {status['deficits']}")
            print(
                {
                    "accepted": len(canonical),
                    "rejected": len(read_jsonl(REJECTED)),
                    "split_counts": pd.Series(
                        [row["split"] for row in canonical]
                    ).value_counts().to_dict(),
                }
            )
            """
        ),
        code(
            """
            grpo = build_grpo_dataset(canonical)
            expected_splits = {
                "train": GRPO_SPLIT_TARGETS["grpo_train"],
                "validation": GRPO_SPLIT_TARGETS["grpo_validation"],
                "test": GRPO_SPLIT_TARGETS["test"],
            }
            actual_splits = {name: len(split) for name, split in grpo.items()}
            if actual_splits != expected_splits:
                raise RuntimeError(
                    f"Unexpected GRPO split sizes: {actual_splits}; expected {expected_splits}"
                )
            if GRPO_DISK.exists():
                shutil.rmtree(GRPO_DISK)
            grpo.save_to_disk(GRPO_DISK)
            for split_name, split_data in grpo.items():
                split_data.to_parquet(
                    DATA / "processed" / f"scientific_design_grpo_{split_name}.parquet"
                )
            if PUSH_DATASETS_TO_HUB:
                grpo.push_to_hub(
                    DATASET_HF_REPO,
                    config_name=DATASET_CONFIG_NAME,
                    private=DATASET_PRIVATE,
                    token=HF_TOKEN,
                    commit_message="Publish scientific design GRPO dataset",
                )
            for split_data in grpo.values():
                assert "source_text" not in split_data.column_names
                assert "completion" not in split_data.column_names
                assert "answer" not in split_data.column_names
            print(grpo)
            """
        ),
        code(
            """
            row = grpo["train"][0]
            display(Markdown("### What the policy sees"))
            display(JSON(row["prompt"]))
            display(Markdown("### What the reward can see"))
            display(
                JSON(
                    {
                        key: row[key]
                        for key in (
                            "task",
                            "required_constraints",
                            "evaluation_criteria",
                            "acceptable_alternatives",
                            "failure_modes",
                        )
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 3. Demonstrate the exact reward

            A malformed response gets `F=0` and therefore total reward zero without an API
            call. A valid response gets the 0.10 format base plus 0.90 times Luna's semantic
            score. Luna grades all four dimensions from 0 to 4 and accepts valid alternatives.
            """
        ),
        code(
            """
            canonical_by_id = {item["task_id"]: item for item in canonical}
            reference_record = canonical_by_id[row["task_id"]]
            reference_completion = render_completion(reference_record)
            format_score = format_reward([reference_completion])[0]

            judge_item = {
                "task": row["task"],
                "required_constraints": row["required_constraints"],
                "evaluation_criteria": row["evaluation_criteria"],
                "acceptable_alternatives": row["acceptable_alternatives"],
                "failure_modes": row["failure_modes"],
                "student_response": reference_completion,
            }
            judge = ScientificDesignJudge(
                model=JUDGE_MODEL,
                cache_path=JUDGE_CACHE,
                api_max_retries=OPENAI_MAX_RETRIES,
                api_timeout_seconds=OPENAI_TIMEOUT_SECONDS,
            )
            judgment = judge.judgments([judge_item])[0]
            semantic_score = semantic_score_from_judgment(judgment)
            total_reward = combined_reward(
                format_score,
                semantic_score,
                format_base_reward=FORMAT_BASE_REWARD,
                semantic_reward_weight=SEMANTIC_REWARD_WEIGHT,
            )
            display(
                JSON(
                    {
                        "format_score_F": format_score,
                        "luna_judgment": judgment,
                        "semantic_score_J": semantic_score,
                        "total_reward_R": total_reward,
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 4. Audit the GRPO curriculum

            The following plots verify family balance, task sizes, and rubric depth. The
            final test split remains untouched by either SFT or GRPO updates.
            """
        ),
        code(
            """
            frame = pd.DataFrame(
                [
                    {
                        "split": split_name,
                        "task_family": item["task_family"],
                        "task_chars": len(item["task"]),
                        "constraints": len(item["required_constraints"]),
                        "criteria": len(item["evaluation_criteria"]),
                    }
                    for split_name, split_data in grpo.items()
                    for item in split_data
                ]
            )
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            sns.countplot(
                frame,
                x="task_family",
                hue="split",
                ax=axes[0],
            )
            axes[0].tick_params(axis="x", rotation=35)
            axes[0].set_title("GRPO task-family balance")
            sns.scatterplot(
                frame,
                x="constraints",
                y="criteria",
                hue="task_family",
                alpha=0.7,
                ax=axes[1],
            )
            axes[1].set_title("Hidden rubric depth")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## Result

            The default GRPO dataset contains 500 training, 75 validation, and 100 final-test
            tasks. Notebook 03 trains the four-section SFT policy; notebook 04 then uses the
            single combined Luna reward demonstrated above.
            """
        ),
    ]
)


nb3 = notebook(
    [
        markdown(
            """
            # 03 — LoRA supervised fine-tuning

            Completion-only SFT teaches Gemma 4 to turn a task-only prompt into the four
            scientific work-product sections. The base weights remain frozen and only LoRA
            parameters are optimized. Device selection is automatic: **CUDA → MPS → CPU**.
            """
        ),
        markdown(
            """
            ## Configuration

            All model, LoRA, training, checkpoint, generation, and Hub settings used below
            are explicit variables. Hugging Face uses cached authentication by default.
            """
        ),
        code(
            """
            import os
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from datasets import load_dataset, load_from_disk
            from IPython.display import JSON, Markdown, display
            from trl import SFTConfig, SFTTrainer

            from science_course.devices import clear_device_cache, detect_runtime
            from science_course.hub import require_hf_namespace
            from science_course.modeling import (
                generate_text,
                load_causal_lm,
                load_tokenizer,
                render_prompt,
                render_sft_completion,
                teaching_lora_config,
            )
            from science_course.versions import require_training_stack

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent

            # Runtime, input source, and local artifacts
            ENABLE_MPS_FALLBACK = True
            TOKENIZERS_PARALLELISM = False
            if ENABLE_MPS_FALLBACK:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = str(TOKENIZERS_PARALLELISM).lower()
            MODEL_ID = "google/gemma-4-E4B-it"
            SFT_DATA_SOURCE_MODE = "hub"  # "hub" or "local"
            LOCAL_SFT_DATA = ROOT / "data" / "processed" / "scientific_design_sft"
            OUTPUT_DIR = ROOT / "artifacts" / "gemma4-scientific-design-sft"
            RESUME_FROM_CHECKPOINT = None
            AUTO_RESUME_LATEST_CHECKPOINT = True

            # LoRA
            LORA_RANK = 16
            LORA_ALPHA = 32
            LORA_DROPOUT = 0.05
            LORA_TARGET_MODULES = "all-linear"

            # SFT optimization
            NUM_TRAIN_EPOCHS = 3
            MAX_STEPS = -1
            LEARNING_RATE = 1e-4
            TRAIN_BATCH_SIZE = 1
            EVAL_BATCH_SIZE = 1
            GRADIENT_ACCUMULATION_STEPS = 8
            MAX_SEQUENCE_LENGTH = 1_024
            MPS_PAD_TO_MULTIPLE_OF = 128
            OTHER_PAD_TO_MULTIPLE_OF = 8
            MPS_EMPTY_CACHE_STEPS = 5
            OTHER_EMPTY_CACHE_STEPS = None
            COMPLETION_ONLY_LOSS = True
            LOSS_TYPE = "nll"
            GRADIENT_CHECKPOINTING = True
            GRADIENT_CHECKPOINTING_USE_REENTRANT = False
            OPTIMIZER = "adamw_torch"
            WARMUP_STEPS = 10
            EVAL_STRATEGY = "steps"
            EVAL_STEPS = 25
            SAVE_STRATEGY = "steps"
            SAVE_STEPS = 25
            SAVE_TOTAL_LIMIT = None
            LOGGING_STEPS = 5
            LOGGING_FIRST_STEP = True
            REPORT_TO = "none"
            RANDOM_SEED = 17
            PIN_MEMORY_ON_CUDA_ONLY = True

            # Inference demonstration
            MAX_NEW_TOKENS = 512
            INFERENCE_DO_SAMPLE = False
            INFERENCE_TEMPERATURE = 1.0
            INFERENCE_TOP_P = 1.0

            # Hugging Face publication
            DATASET_HF_REPO = "lamm-mit/scientific-sft-grpo-data"
            DATASET_CONFIG_NAME = "scientific_design_sft"
            SFT_HF_REPO = "lamm-mit/scientific-sft-grpo-design-sft"
            PUSH_TO_HUB = True
            HUB_STRATEGY = "all_checkpoints"
            HUB_PRIVATE_REPO = False
            HUB_ALWAYS_PUSH = True
            HF_TOKEN = None
            # HF_TOKEN = os.environ["HF_TOKEN"]  # Optional; prefer `hf auth login`.

            versions = require_training_stack()
            runtime = detect_runtime()
            if SFT_DATA_SOURCE_MODE not in {"hub", "local"}:
                raise ValueError("SFT_DATA_SOURCE_MODE must be 'hub' or 'local'.")
            DATALOADER_PIN_MEMORY = (
                runtime.backend == "cuda" if PIN_MEMORY_ON_CUDA_ONLY else True
            )
            PAD_TO_MULTIPLE_OF = (
                MPS_PAD_TO_MULTIPLE_OF
                if runtime.backend == "mps"
                else OTHER_PAD_TO_MULTIPLE_OF
            )
            TORCH_EMPTY_CACHE_STEPS = (
                MPS_EMPTY_CACHE_STEPS
                if runtime.backend == "mps"
                else OTHER_EMPTY_CACHE_STEPS
            )
            if AUTO_RESUME_LATEST_CHECKPOINT and RESUME_FROM_CHECKPOINT is None:
                checkpoints = sorted(
                    OUTPUT_DIR.glob("checkpoint-*"),
                    key=lambda path: int(path.name.rsplit("-", 1)[-1]),
                )
                if checkpoints:
                    RESUME_FROM_CHECKPOINT = str(checkpoints[-1])
            if PUSH_TO_HUB:
                require_hf_namespace(SFT_HF_REPO, token=HF_TOKEN)
            display(
                JSON(
                    {
                        "runtime": runtime.as_dict(),
                        "versions": versions,
                        "model": MODEL_ID,
                        "dataset_source_mode": SFT_DATA_SOURCE_MODE,
                        "local_dataset": str(LOCAL_SFT_DATA),
                        "hub_dataset": f"{DATASET_HF_REPO}/{DATASET_CONFIG_NAME}",
                        "hub_model": SFT_HF_REPO,
                        "max_sequence_length": MAX_SEQUENCE_LENGTH,
                        "pad_to_multiple_of": PAD_TO_MULTIPLE_OF,
                        "torch_empty_cache_steps": TORCH_EMPTY_CACHE_STEPS,
                        "resume_from_checkpoint": RESUME_FROM_CHECKPOINT,
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 1. Render task-only prompts and structured completions

            The tokenizer's chat template formats each prompt. Loss is computed only on the
            assistant completion, which contains the four tagged sections.
            """
        ),
        code(
            """
            if SFT_DATA_SOURCE_MODE == "hub":
                sft = load_dataset(
                    DATASET_HF_REPO,
                    DATASET_CONFIG_NAME,
                    token=HF_TOKEN,
                )
            else:
                if not LOCAL_SFT_DATA.exists():
                    raise RuntimeError(
                        "LOCAL_SFT_DATA does not exist. Run notebook 01 or use "
                        "SFT_DATA_SOURCE_MODE='hub'."
                    )
                sft = load_from_disk(LOCAL_SFT_DATA)
            if len(sft["train"]) == 0 or len(sft["validation"]) == 0:
                raise RuntimeError("Both SFT train and validation splits must be non-empty.")

            tokenizer = load_tokenizer(MODEL_ID, token=HF_TOKEN)

            def render_row(row):
                return {
                    "prompt": render_prompt(tokenizer, row["prompt"]),
                    "completion": render_sft_completion(tokenizer, row["completion"]),
                }

            rendered = sft.map(
                render_row,
                remove_columns=sft["train"].column_names,
                desc="Apply the Gemma chat template",
            )
            display(JSON(rendered["train"][0]))
            """
        ),
        code(
            """
            lengths = pd.DataFrame(
                {
                    split_name: pd.Series(
                        [
                            len(
                                tokenizer(
                                    item["prompt"] + item["completion"]
                                ).input_ids
                            )
                            for item in split_data
                        ]
                    )
                    for split_name, split_data in rendered.items()
                }
            )
            lengths.plot.hist(bins=25, alpha=0.65, figsize=(10, 4.5))
            plt.axvline(
                MAX_SEQUENCE_LENGTH,
                color="crimson",
                linestyle="--",
                label="training max_length",
            )
            plt.xlabel("tokens")
            plt.title("Structured SFT sequence-length audit")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 2. Attach LoRA and configure the trainer

            No CUDA-only quantization dependency is used, so the same path works on CUDA,
            Apple MPS, and CPU.
            """
        ),
        code(
            """
            clear_device_cache(runtime)
            model = load_causal_lm(MODEL_ID, runtime, token=HF_TOKEN)
            lora = teaching_lora_config(
                rank=LORA_RANK,
                alpha=LORA_ALPHA,
                dropout=LORA_DROPOUT,
                target_modules=LORA_TARGET_MODULES,
            )

            sft_args = SFTConfig(
                output_dir=str(OUTPUT_DIR),
                num_train_epochs=NUM_TRAIN_EPOCHS,
                max_steps=MAX_STEPS,
                learning_rate=LEARNING_RATE,
                per_device_train_batch_size=TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=EVAL_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                max_length=MAX_SEQUENCE_LENGTH,
                pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
                completion_only_loss=COMPLETION_ONLY_LOSS,
                loss_type=LOSS_TYPE,
                gradient_checkpointing=GRADIENT_CHECKPOINTING,
                gradient_checkpointing_kwargs={
                    "use_reentrant": GRADIENT_CHECKPOINTING_USE_REENTRANT
                },
                optim=OPTIMIZER,
                warmup_steps=WARMUP_STEPS,
                eval_strategy=EVAL_STRATEGY,
                eval_steps=EVAL_STEPS,
                save_strategy=SAVE_STRATEGY,
                save_steps=SAVE_STEPS,
                save_total_limit=SAVE_TOTAL_LIMIT,
                logging_steps=LOGGING_STEPS,
                logging_first_step=LOGGING_FIRST_STEP,
                report_to=REPORT_TO,
                torch_empty_cache_steps=TORCH_EMPTY_CACHE_STEPS,
                dataloader_pin_memory=DATALOADER_PIN_MEMORY,
                push_to_hub=PUSH_TO_HUB,
                hub_model_id=SFT_HF_REPO,
                hub_strategy=HUB_STRATEGY,
                hub_private_repo=HUB_PRIVATE_REPO,
                hub_token=HF_TOKEN,
                hub_always_push=HUB_ALWAYS_PUSH,
                bf16=runtime.trainer_bf16,
                fp16=runtime.trainer_fp16,
                use_cpu=runtime.use_cpu,
                seed=RANDOM_SEED,
            )
            trainer = SFTTrainer(
                model=model,
                args=sft_args,
                train_dataset=rendered["train"],
                eval_dataset=rendered["validation"],
                processing_class=tokenizer,
                peft_config=lora,
            )
            trainer.model.print_trainable_parameters()
            """
        ),
        markdown(
            """
            ## 3. Train, evaluate, checkpoint, and publish

            Every saved checkpoint and the final adapter are uploaded when publication is
            enabled.
            """
        ),
        code(
            """
            train_result = trainer.train(
                resume_from_checkpoint=RESUME_FROM_CHECKPOINT
            )
            trainer.save_model(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            if PUSH_TO_HUB:
                hub_result = trainer.push_to_hub(
                    commit_message="Complete scientific design SFT training"
                )
                print(f"Published final adapter and all checkpoints: {hub_result}")
            display(JSON(train_result.metrics))
            """
        ),
        code(
            """
            history = pd.DataFrame(trainer.state.log_history)
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            if "loss" in history:
                history.dropna(subset=["loss"]).plot(
                    x="step", y="loss", ax=axes[0], color="#315c8c", legend=False
                )
            axes[0].set_title("Completion-only SFT loss")
            if "eval_loss" in history:
                history.dropna(subset=["eval_loss"]).plot(
                    x="step",
                    y="eval_loss",
                    ax=axes[1],
                    color="#d97732",
                    legend=False,
                )
            axes[1].set_title("Held-out SFT loss")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 4. Use the adapter on a new task

            Inference needs only a new task. It does not require a paper, reference completion,
            hidden rubric, or teacher API call.
            """
        ),
        code(
            """
            new_task = (
                "Design a self-healing hydrogel for repeated deformation in water. "
                "The material may use reversible physical interactions, but recovery must "
                "not require external heating. Develop several mechanistic strategies, "
                "identify the governing design principles, synthesize the strongest design, "
                "and give a final recommendation."
            )
            new_messages = [
                {
                    "role": "system",
                    "content": (
                        "Solve the self-contained scientific problem-solving task. Develop "
                        "candidate ideas, identify principles, synthesize them, and answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"SCIENTIFIC PROBLEM-SOLVING TASK\\n{new_task}\\n\\n"
                        "Respond with <brainstorm>, <principles>, <synthesis>, and "
                        "<answer> in that order, with no text outside the tags."
                    ),
                },
            ]
            prompt = render_prompt(tokenizer, new_messages)
            response = generate_text(
                trainer.model,
                tokenizer,
                prompt,
                runtime,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=INFERENCE_DO_SAMPLE,
                temperature=INFERENCE_TEMPERATURE,
                top_p=INFERENCE_TOP_P,
            )
            display(Markdown(f"```text\\n{response}\\n```"))
            """
        ),
        markdown(
            """
            ## Result

            The local adapter is in `artifacts/gemma4-scientific-design-sft/`, and the final
            adapter plus every saved checkpoint are published to the configured Hub model
            repository. Notebook 04 continues this same adapter with Luna-judged GRPO.
            """
        ),
    ]
)


nb4 = notebook(
    [
        markdown(
            """
            # 04 — Continue the SFT adapter with GRPO

            For each task, GRPO samples four structured completions, scores them, and reinforces
            completions that are better relative to the group. The reward stays intentionally
            simple:

            \\[
            R = F\\times(0.10 + 0.90J)
            \\]

            `F` is the exact format gate. `J` is `gpt-5.6-luna`'s normalized four-dimension
            scientific judgment.
            """
        ),
        markdown(
            """
            ## Important execution note

            The semantic reward is live API work because the policy creates new completions
            during training. The cache makes reruns resumable. This notebook intentionally
            requires one training process; a distributed production run should use a
            concurrency-safe reward service.
            """
        ),
        markdown(
            """
            ## Configuration

            Every model, reward coefficient, sampling setting, optimizer parameter, checkpoint
            setting, path, and Hub repository used below is an explicit variable.
            """
        ),
        code(
            """
            import os
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from accelerate import PartialState
            from datasets import load_dataset, load_from_disk
            from IPython.display import JSON, Markdown, display
            from peft import PeftModel
            from trl import GRPOConfig, GRPOTrainer

            from science_course.devices import clear_device_cache, detect_runtime
            from science_course.hub import require_hf_namespace
            from science_course.judge import (
                DEFAULT_JUDGE_MODEL,
                configure_scientific_design_judge,
                scientific_design_reward,
            )
            from science_course.modeling import (
                generate_text,
                load_causal_lm,
                load_tokenizer,
                render_prompt,
            )
            from science_course.teacher import require_openai_key
            from science_course.versions import require_training_stack

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent

            # Runtime, input sources, and local artifacts
            ENABLE_MPS_FALLBACK = True
            TOKENIZERS_PARALLELISM = False
            if ENABLE_MPS_FALLBACK:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = str(TOKENIZERS_PARALLELISM).lower()
            MODEL_ID = "google/gemma-4-E4B-it"
            GRPO_DATA_SOURCE_MODE = "hub"  # "hub" or "local"
            SFT_ADAPTER_SOURCE_MODE = "hub"  # "hub" or "local"
            LOCAL_SFT_ADAPTER = ROOT / "artifacts" / "gemma4-scientific-design-sft"
            LOCAL_GRPO_DATA = ROOT / "data" / "processed" / "scientific_design_grpo"
            GRPO_ADAPTER = ROOT / "artifacts" / "gemma4-scientific-design-grpo"
            RESUME_FROM_CHECKPOINT = None

            # Exact Luna reward
            JUDGE_MODEL = DEFAULT_JUDGE_MODEL
            JUDGE_CACHE = ROOT / "results" / "scientific_design_judge_cache.jsonl"
            FORMAT_BASE_REWARD = 0.10
            SEMANTIC_REWARD_WEIGHT = 0.90
            REWARD_FUNCTION_WEIGHTS = [1.0]
            OPENAI_MAX_RETRIES = 3
            OPENAI_TIMEOUT_SECONDS = 120.0

            # GRPO optimization and sampling
            NUM_TRAIN_EPOCHS = 1
            MAX_STEPS = -1
            LEARNING_RATE = 5e-6
            TRAIN_BATCH_SIZE = 1
            EVAL_BATCH_SIZE = 4
            GRADIENT_ACCUMULATION_STEPS = 4
            NUM_GENERATIONS = 4
            NUM_GENERATIONS_EVAL = 4
            MAX_COMPLETION_LENGTH = 512
            TEMPERATURE = 0.8
            TOP_P = 0.95
            TOP_K = 0
            BETA = 0.0
            SCALE_REWARDS = "group"
            MULTI_OBJECTIVE_AGGREGATION = "sum_then_normalize"
            LOSS_TYPE = "dapo"
            GRADIENT_CHECKPOINTING = True
            GRADIENT_CHECKPOINTING_USE_REENTRANT = False
            OPTIMIZER = "adamw_torch"
            WARMUP_STEPS = 5
            # Full GRPO evaluation is expensive: 75 tasks × 4 rollouts = 300
            # candidate responses plus Luna grading. With one epoch, "epoch" runs it once.
            # For periodic evaluation instead, use "steps" and set EVAL_STEPS (for example 250).
            EVAL_STRATEGY = "epoch"
            EVAL_STEPS = None
            SAVE_STRATEGY = "steps"
            SAVE_STEPS = 25
            SAVE_TOTAL_LIMIT = None
            LOGGING_STEPS = 1
            LOGGING_FIRST_STEP = True
            LOG_COMPLETIONS = True
            NUM_COMPLETIONS_TO_PRINT = 4
            REPORT_TO = "none"
            USE_VLLM = False
            RANDOM_SEED = 17
            PIN_MEMORY_ON_CUDA_ONLY = True

            # Before/after inference comparison
            MAX_NEW_TOKENS = 512
            INFERENCE_DO_SAMPLE = False
            INFERENCE_TEMPERATURE = 1.0
            INFERENCE_TOP_P = 1.0

            # Hugging Face publication
            DATASET_HF_REPO = "lamm-mit/scientific-sft-grpo-data"
            DATASET_CONFIG_NAME = "scientific_design_grpo"
            SFT_HF_REPO = "lamm-mit/scientific-sft-grpo-design-sft"
            GRPO_HF_REPO = "lamm-mit/scientific-sft-grpo-design-grpo"
            PUSH_TO_HUB = True
            HUB_STRATEGY = "all_checkpoints"
            HUB_PRIVATE_REPO = False
            HUB_ALWAYS_PUSH = True
            HF_TOKEN = None
            # HF_TOKEN = os.environ["HF_TOKEN"]  # Optional; prefer `hf auth login`.

            require_openai_key()
            versions = require_training_stack()
            runtime = detect_runtime()
            if GRPO_DATA_SOURCE_MODE not in {"hub", "local"}:
                raise ValueError("GRPO_DATA_SOURCE_MODE must be 'hub' or 'local'.")
            if SFT_ADAPTER_SOURCE_MODE not in {"hub", "local"}:
                raise ValueError("SFT_ADAPTER_SOURCE_MODE must be 'hub' or 'local'.")
            SFT_ADAPTER_SOURCE = (
                SFT_HF_REPO
                if SFT_ADAPTER_SOURCE_MODE == "hub"
                else LOCAL_SFT_ADAPTER
            )
            DATALOADER_PIN_MEMORY = (
                runtime.backend == "cuda" if PIN_MEMORY_ON_CUDA_ONLY else True
            )
            if JUDGE_MODEL != "gpt-5.6-luna":
                raise RuntimeError("The configured semantic judge must be gpt-5.6-luna.")
            if abs(FORMAT_BASE_REWARD + SEMANTIC_REWARD_WEIGHT - 1.0) > 1e-9:
                raise ValueError("Reward coefficients must sum to one.")
            if EVAL_BATCH_SIZE % NUM_GENERATIONS_EVAL != 0:
                raise ValueError(
                    "EVAL_BATCH_SIZE must be divisible by NUM_GENERATIONS_EVAL."
                )
            if PartialState().num_processes != 1:
                raise RuntimeError("This API-judged teaching run requires WORLD_SIZE=1.")
            configure_scientific_design_judge(
                model=JUDGE_MODEL,
                cache_path=JUDGE_CACHE,
                format_base_reward=FORMAT_BASE_REWARD,
                semantic_reward_weight=SEMANTIC_REWARD_WEIGHT,
                api_max_retries=OPENAI_MAX_RETRIES,
                api_timeout_seconds=OPENAI_TIMEOUT_SECONDS,
            )
            if PUSH_TO_HUB:
                require_hf_namespace(GRPO_HF_REPO, token=HF_TOKEN)
            display(
                JSON(
                    {
                        "runtime": runtime.as_dict(),
                        "versions": versions,
                        "model": MODEL_ID,
                        "grpo_dataset_source_mode": GRPO_DATA_SOURCE_MODE,
                        "sft_adapter_source_mode": SFT_ADAPTER_SOURCE_MODE,
                        "sft_adapter": str(SFT_ADAPTER_SOURCE),
                        "judge_model": JUDGE_MODEL,
                        "reward_formula": "F * (0.10 + 0.90 * J)",
                        "hub_model": GRPO_HF_REPO,
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 1. Load the task-only GRPO data and trainable SFT adapter

            The policy receives only `prompt`. TRL passes the hidden rubric columns to the
            reward function.
            """
        ),
        code(
            """
            if GRPO_DATA_SOURCE_MODE == "hub":
                grpo = load_dataset(
                    DATASET_HF_REPO,
                    DATASET_CONFIG_NAME,
                    token=HF_TOKEN,
                )
            else:
                if not LOCAL_GRPO_DATA.exists():
                    raise RuntimeError(
                        "LOCAL_GRPO_DATA does not exist. Run notebook 02 or use "
                        "GRPO_DATA_SOURCE_MODE='hub'."
                    )
                grpo = load_from_disk(LOCAL_GRPO_DATA)
            if (
                SFT_ADAPTER_SOURCE_MODE == "local"
                and not LOCAL_SFT_ADAPTER.exists()
            ):
                raise RuntimeError(
                    "LOCAL_SFT_ADAPTER does not exist. Run notebook 03 or use "
                    "SFT_ADAPTER_SOURCE_MODE='hub'."
                )
            if len(grpo["train"]) == 0 or len(grpo["validation"]) == 0:
                raise RuntimeError("GRPO train and validation splits must be non-empty.")

            tokenizer = load_tokenizer(MODEL_ID, token=HF_TOKEN)
            clear_device_cache(runtime)
            base_model = load_causal_lm(MODEL_ID, runtime, token=HF_TOKEN)
            model = PeftModel.from_pretrained(
                base_model,
                SFT_ADAPTER_SOURCE,
                is_trainable=True,
                token=HF_TOKEN,
            )
            model.print_trainable_parameters()
            print(grpo)
            """
        ),
        markdown(
            """
            ## 2. Snapshot the SFT policy on one unseen test task

            This qualitative before/after comparison is useful for teaching but is not a
            substitute for aggregate expert evaluation.
            """
        ),
        code(
            """
            held_out = grpo["test"][0]
            held_out_prompt = render_prompt(tokenizer, held_out["prompt"])
            before_grpo = generate_text(
                model,
                tokenizer,
                held_out_prompt,
                runtime,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=INFERENCE_DO_SAMPLE,
                temperature=INFERENCE_TEMPERATURE,
                top_p=INFERENCE_TOP_P,
            )
            display(Markdown("### SFT policy\\n```text\\n" + before_grpo + "\\n```"))
            """
        ),
        markdown(
            """
            ## 3. Configure the exact combined reward

            For each completion:

            1. `F=1` only when all four non-empty sections appear in exact order; otherwise
               `F=0` and the reward is zero without calling Luna.
            2. Luna assigns integer 0–4 scores for brainstorm, principles, synthesis, and
               answer.
            3. `J=(B+P+S+A)/16`.
            4. `R=F×(0.10+0.90J)`.

            GRPO then normalizes these raw rewards within each four-completion group.
            """
        ),
        code(
            """
            grpo_args = GRPOConfig(
                output_dir=str(GRPO_ADAPTER),
                num_train_epochs=NUM_TRAIN_EPOCHS,
                max_steps=MAX_STEPS,
                learning_rate=LEARNING_RATE,
                per_device_train_batch_size=TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=EVAL_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                num_generations=NUM_GENERATIONS,
                num_generations_eval=NUM_GENERATIONS_EVAL,
                max_completion_length=MAX_COMPLETION_LENGTH,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                beta=BETA,
                reward_weights=REWARD_FUNCTION_WEIGHTS,
                scale_rewards=SCALE_REWARDS,
                multi_objective_aggregation=MULTI_OBJECTIVE_AGGREGATION,
                loss_type=LOSS_TYPE,
                remove_unused_columns=False,
                chat_template_kwargs={"enable_thinking": False},
                gradient_checkpointing=GRADIENT_CHECKPOINTING,
                gradient_checkpointing_kwargs={
                    "use_reentrant": GRADIENT_CHECKPOINTING_USE_REENTRANT
                },
                optim=OPTIMIZER,
                warmup_steps=WARMUP_STEPS,
                eval_strategy=EVAL_STRATEGY,
                eval_steps=EVAL_STEPS,
                save_strategy=SAVE_STRATEGY,
                save_steps=SAVE_STEPS,
                save_total_limit=SAVE_TOTAL_LIMIT,
                logging_steps=LOGGING_STEPS,
                logging_first_step=LOGGING_FIRST_STEP,
                log_completions=LOG_COMPLETIONS,
                num_completions_to_print=NUM_COMPLETIONS_TO_PRINT,
                report_to=REPORT_TO,
                dataloader_pin_memory=DATALOADER_PIN_MEMORY,
                push_to_hub=PUSH_TO_HUB,
                hub_model_id=GRPO_HF_REPO,
                hub_strategy=HUB_STRATEGY,
                hub_private_repo=HUB_PRIVATE_REPO,
                hub_token=HF_TOKEN,
                hub_always_push=HUB_ALWAYS_PUSH,
                bf16=runtime.trainer_bf16,
                fp16=runtime.trainer_fp16,
                use_cpu=runtime.use_cpu,
                use_vllm=USE_VLLM,
                seed=RANDOM_SEED,
            )
            trainer = GRPOTrainer(
                model=model,
                args=grpo_args,
                reward_funcs=[scientific_design_reward],
                train_dataset=grpo["train"],
                eval_dataset=grpo["validation"],
                processing_class=tokenizer,
            )
            """
        ),
        markdown(
            """
            ## 4. Train, evaluate, checkpoint, and publish

            The Luna cache is append-only and content-addressed, so completed judgments are
            reused after interruption. Every saved trainer checkpoint is published.

            With the default `EVAL_STRATEGY="epoch"` and one training epoch, held-out
            evaluation runs once at the end. It generates four fresh rollouts for each of the
            75 validation tasks and applies the same reward function—up to 300 Luna judgments.
            These `eval_*` rewards are diagnostics and do not update the policy. The
            unprefixed training reward is the signal used by GRPO.
            """
        ),
        code(
            """
            train_result = trainer.train(
                resume_from_checkpoint=RESUME_FROM_CHECKPOINT
            )
            trainer.save_model(GRPO_ADAPTER)
            tokenizer.save_pretrained(GRPO_ADAPTER)
            if PUSH_TO_HUB:
                hub_result = trainer.push_to_hub(
                    commit_message="Complete scientific design GRPO training"
                )
                print(f"Published final adapter and all checkpoints: {hub_result}")
            display(JSON(train_result.metrics))
            """
        ),
        code(
            """
            history = pd.DataFrame(trainer.state.log_history)
            training_reward_columns = [
                column
                for column in history.columns
                if column == "reward"
                or (column.startswith("rewards/") and column.endswith("/mean"))
            ]
            evaluation_reward_columns = [
                column
                for column in history.columns
                if column == "eval_reward"
                or (
                    column.startswith("eval_rewards/")
                    and column.endswith("/mean")
                )
            ]
            reward_columns = training_reward_columns + evaluation_reward_columns
            if reward_columns:
                history[["step", *reward_columns]].dropna(
                    how="all",
                    subset=reward_columns,
                ).plot(
                    x="step",
                    y=reward_columns,
                    figsize=(12, 5),
                    marker="o",
                )
                plt.title("Training reward and held-out evaluation reward")
                plt.ylabel("reward")
                plt.tight_layout()
                plt.show()
            else:
                print("Reward columns available:", sorted(history.columns))
            """
        ),
        markdown(
            """
            ## 5. Compare the same unseen task

            Inspect whether the GRPO policy develops more distinct candidates, states the
            relevant constraints, synthesizes rather than lists, and provides a defensible
            final answer.
            """
        ),
        code(
            """
            after_grpo = generate_text(
                trainer.model,
                tokenizer,
                held_out_prompt,
                runtime,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=INFERENCE_DO_SAMPLE,
                temperature=INFERENCE_TEMPERATURE,
                top_p=INFERENCE_TOP_P,
            )
            display(Markdown("### Before GRPO\\n```text\\n" + before_grpo + "\\n```"))
            display(Markdown("### After GRPO\\n```text\\n" + after_grpo + "\\n```"))
            display(Markdown("### Hidden instructor rubric"))
            display(
                JSON(
                    {
                        "required_constraints": held_out["required_constraints"],
                        "evaluation_criteria": held_out["evaluation_criteria"],
                        "acceptable_alternatives": held_out["acceptable_alternatives"],
                        "failure_modes": held_out["failure_modes"],
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 6. Fresh-kernel inference from Hugging Face

            The preceding comparison intentionally uses the in-memory trainer. This cell is
            independent of that state: it can be run after reopening the notebook or in a
            fresh kernel. By default it loads the final adapter from the root of the GRPO Hub
            repository. Set `INFERENCE_ADAPTER_SUBFOLDER` to a published `checkpoint-*`
            directory, or set `INFERENCE_ADAPTER_REVISION` to a branch, tag, or commit.

            Inference requires Hugging Face access to gated Gemma weights but does **not**
            require an OpenAI key or a Luna call.
            """
        ),
        code(
            """
            import gc
            import os

            from IPython.display import Markdown, display
            from peft import PeftModel

            from science_course.data import task_prompt
            from science_course.devices import clear_device_cache, detect_runtime
            from science_course.modeling import (
                generate_text,
                load_causal_lm,
                load_tokenizer,
                render_prompt,
            )

            # All fresh-kernel inference settings are explicit here.
            INFERENCE_BASE_MODEL_ID = "google/gemma-4-E4B-it"
            INFERENCE_ADAPTER_REPO = "lamm-mit/scientific-sft-grpo-design-grpo"
            INFERENCE_ADAPTER_REVISION = None  # Optional branch, tag, or commit hash.
            INFERENCE_ADAPTER_SUBFOLDER = None  # Example: "checkpoint-100".
            INFERENCE_MAX_NEW_TOKENS = 512
            INFERENCE_DO_SAMPLE = False
            INFERENCE_TEMPERATURE = 1.0
            INFERENCE_TOP_P = 1.0
            INFERENCE_HF_TOKEN = None
            # INFERENCE_HF_TOKEN = os.environ["HF_TOKEN"]  # Optional; cached login is preferred.

            INFERENCE_TASK = (
                "Design a self-healing hydrogel for repeated deformation in water. "
                "The material may use reversible physical interactions, but recovery must "
                "not require external heating. Develop several mechanistic strategies, "
                "identify the governing design principles, synthesize the strongest design, "
                "and give a final recommendation."
            )

            # Release any prior trainer/model objects if this cell follows training.
            for object_name in ("trainer", "model", "base_model"):
                globals().pop(object_name, None)
            gc.collect()

            inference_runtime = detect_runtime()
            clear_device_cache(inference_runtime)
            inference_tokenizer = load_tokenizer(
                INFERENCE_BASE_MODEL_ID,
                token=INFERENCE_HF_TOKEN,
            )
            inference_base_model = load_causal_lm(
                INFERENCE_BASE_MODEL_ID,
                inference_runtime,
                token=INFERENCE_HF_TOKEN,
            )
            adapter_load_kwargs = {
                "is_trainable": False,
                "token": INFERENCE_HF_TOKEN,
            }
            if INFERENCE_ADAPTER_REVISION is not None:
                adapter_load_kwargs["revision"] = INFERENCE_ADAPTER_REVISION
            if INFERENCE_ADAPTER_SUBFOLDER is not None:
                adapter_load_kwargs["subfolder"] = INFERENCE_ADAPTER_SUBFOLDER
            inference_model = PeftModel.from_pretrained(
                inference_base_model,
                INFERENCE_ADAPTER_REPO,
                **adapter_load_kwargs,
            )

            inference_prompt = render_prompt(
                inference_tokenizer,
                task_prompt(INFERENCE_TASK),
            )
            inference_response = generate_text(
                inference_model,
                inference_tokenizer,
                inference_prompt,
                inference_runtime,
                max_new_tokens=INFERENCE_MAX_NEW_TOKENS,
                do_sample=INFERENCE_DO_SAMPLE,
                temperature=INFERENCE_TEMPERATURE,
                top_p=INFERENCE_TOP_P,
            )
            display(
                Markdown(
                    "### Reloaded GRPO adapter response\\n"
                    f"Repository: `{INFERENCE_ADAPTER_REPO}`  \\n"
                    f"Subfolder: `{INFERENCE_ADAPTER_SUBFOLDER or 'final adapter'}`\\n\\n"
                    f"```text\\n{inference_response}\\n```"
                )
            )
            """
        ),
        markdown(
            """
            ## Result and research caveat

            The final adapter is in `artifacts/gemma4-scientific-design-grpo/` and on the
            configured Hub repository with all saved checkpoints. For research claims, add
            expert grading, inter-rater agreement, multiple seeds, reward ablations,
            contamination checks, and domain-specific held-out evaluations. Luna provides
            scalable supervision—not scientific ground truth.
            """
        ),
    ]
)


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "01_generate_scientific_design_sft_dataset.ipynb": nb1,
        "02_generate_scientific_design_grpo_dataset.ipynb": nb2,
        "03_finetune_sft_lora.ipynb": nb3,
        "04_finetune_grpo_lora.ipynb": nb4,
    }
    for name, artifact in outputs.items():
        path = NOTEBOOKS / name
        if path.exists():
            existing = nbf.read(path, as_version=4)
            existing_ids = {
                (cell.cell_type, cell.source): cell.id
                for cell in existing.cells
                if cell.get("id")
            }
            for cell in artifact.cells:
                prior_id = existing_ids.get((cell.cell_type, cell.source))
                if prior_id is not None:
                    cell.id = prior_id
        nbf.write(artifact, path)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
