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
                "display_name": "Python 3 (scientific-sft-grpo-course)",
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
            # 01 — Build an SFT dataset for scientific mechanisms

            **Learning goal.** Turn openly licensed scientific source material into
            self-contained *how/why* tasks. A teacher model drafts each task; a separate
            critic call checks it against the source. The student never sees the source.

            ```
            open scientific source
                    │  teacher + critic (gpt-5.6-terra)
                    ▼
            self-contained mechanism task
                    │
                    ├── visible causal explanation
                    ├── evidence quoted from the task
                    └── concise mechanistic answer
            ```

            This course deliberately excludes calculation and numerical prediction tasks.
            The target is a causal chain: **condition → intermediate process → outcome**.
            """
        ),
        markdown(
            """
            ## Before you run

            From the repository root:

            ```bash
            python3.12 -m venv .venv
            source .venv/bin/activate
            python -m pip install --upgrade pip
            python -m pip install -e ".[dev]"
            export OPENAI_API_KEY="..."
            hf auth login
            jupyter lab
            ```

            The generation is real, paid API use—there is no offline substitute. Each source
            normally uses two calls (teacher and critic). Results are appended incrementally,
            so interrupting and rerunning is safe.
            """
        ),
        markdown(
            """
            ## Configuration

            Every editable setting is defined below. The OpenAI key is the only required
            environment variable. Hugging Face uses `hf auth login` by default; the commented
            `HF_TOKEN` line is an optional alternative and should never contain a pasted token.
            """
        ),
        code(
            """
            import os
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            from IPython.display import JSON, Markdown, display

            from science_course.data import (
                build_sft_dataset,
                read_jsonl,
                stream_open_science_sources,
                write_jsonl,
            )
            from science_course.hub import require_hf_namespace
            from science_course.teacher import (
                DEFAULT_TEACHER_MODEL,
                generate_canonical_tasks,
                require_openai_key,
            )

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            DATA = ROOT / "data"
            RAW_SOURCES = DATA / "raw" / "open_science_sources.jsonl"
            ACCEPTED = DATA / "canonical" / "mechanism_tasks.jsonl"
            REJECTED = DATA / "canonical" / "rejected_tasks.jsonl"
            SFT_DISK = DATA / "processed" / "sft"

            # Dataset authoring
            TEACHER_MODEL = DEFAULT_TEACHER_MODEL
            SOURCE_DATASET_ID = "common-pile/peS2o"
            SOURCE_SPLIT = "train"
            MAX_SOURCE_PAPERS = 120
            MAX_SOURCE_RECORDS_SCANNED = 20_000
            MIN_SOURCE_CHARS = 1_200
            MAX_SOURCE_CHARS = 6_000
            RANDOM_SEED = 17

            # Hugging Face publication
            DATASET_HF_REPO = "lamm-mit/scientific-sft-grpo-data"
            PUSH_DATASETS_TO_HUB = True
            DATASET_PRIVATE = False
            HF_TOKEN = None
            # HF_TOKEN = os.environ["HF_TOKEN"]  # Optional; prefer `hf auth login`.

            require_openai_key()
            assert TEACHER_MODEL == "gpt-5.6-terra", (
                "This class notebook is tested with the requested teacher model: "
                "gpt-5.6-terra"
            )
            if PUSH_DATASETS_TO_HUB:
                require_hf_namespace(DATASET_HF_REPO, token=HF_TOKEN)

            sns.set_theme(style="whitegrid", context="talk")
            print(
                {
                    "root": str(ROOT),
                    "teacher_model": TEACHER_MODEL,
                    "sources": MAX_SOURCE_PAPERS,
                    "dataset_hub_repo": DATASET_HF_REPO,
                }
            )
            """
        ),
        markdown(
            """
            ## 1. Acquire open source material

            We stream `common-pile/peS2o`, derived from openly licensed scientific papers.
            Per-document license metadata is filtered before any teacher call. Only a bounded
            source excerpt is sent to the teacher; the record retains its paper ID, URL,
            license, split, and content hash for provenance.

            This source text is **authoring material**, not model input after fine-tuning.
            """
        ),
        code(
            """
            cached_sources = read_jsonl(RAW_SOURCES)
            if len(cached_sources) == MAX_SOURCE_PAPERS:
                sources = cached_sources
                print(f"Reusing {len(sources)} source records from {RAW_SOURCES}")
            else:
                sources = stream_open_science_sources(
                    dataset_id=SOURCE_DATASET_ID,
                    split=SOURCE_SPLIT,
                    max_papers=MAX_SOURCE_PAPERS,
                    max_scanned=MAX_SOURCE_RECORDS_SCANNED,
                    min_chars=MIN_SOURCE_CHARS,
                    max_chars=MAX_SOURCE_CHARS,
                    seed=RANDOM_SEED,
                )
                write_jsonl(RAW_SOURCES, sources)
                print(
                    f"Refreshed the source cache with {len(sources)} records at {RAW_SOURCES}"
                )

            if not sources:
                raise RuntimeError(
                    "No open sources passed the filters. Increase MAX_SOURCE_PAPERS and "
                    "rerun acquisition."
                )

            source_frame = pd.DataFrame(sources)
            display(
                source_frame[
                    ["paper_id", "title", "source_license", "split", "source_url"]
                ].head()
            )
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            source_frame["source_license"].value_counts().plot.bar(
                ax=axes[0], color="#315c8c", title="Open-license provenance"
            )
            source_frame["split"].value_counts().reindex(
                ["sft_train", "sft_validation", "grpo_train", "grpo_validation", "test"]
            ).plot.bar(ax=axes[1], color="#d97732", title="Paper-level split")
            for ax in axes:
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=35)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 2. Author and critique mechanism tasks

            The teacher produces a structured record. It must:

            - ask a qualitative **how/why** question;
            - include every observation needed to answer;
            - provide an ordered causal rubric and explicit cause→effect links;
            - quote evidence from the newly written task itself; and
            - avoid arithmetic, numerical targets, and unsupported claims.

            A second call sees both the source and draft and can reject it. Rejections are
            useful audit data, not silently discarded failures.
            """
        ),
        code(
            """
            canonical_all = generate_canonical_tasks(
                sources,
                accepted_path=ACCEPTED,
                rejected_path=REJECTED,
                model=TEACHER_MODEL,
            )
            active_paper_ids = {row["paper_id"] for row in sources}
            canonical = [
                row for row in canonical_all if row["paper_id"] in active_paper_ids
            ]
            rejected = [
                row
                for row in read_jsonl(REJECTED)
                if row.get("paper_id") in active_paper_ids
            ]
            print(
                {
                    "accepted": len(canonical),
                    "rejected": len(rejected),
                    "acceptance_rate": len(canonical) / max(len(canonical) + len(rejected), 1),
                }
            )

            if not canonical:
                raise RuntimeError("No tasks were accepted. Inspect the rejection audit file.")
            """
        ),
        code(
            """
            example = canonical[0]
            display(Markdown("### Student-visible task"))
            display(Markdown(example["task"]))
            display(Markdown("### Reference response"))
            display(
                Markdown(
                    f"**Reasoning:** {example['reasoning']}\\n\\n"
                    f"**Evidence:** “{example['evidence']}”\\n\\n"
                    f"**Answer:** {example['answer']}"
                )
            )
            display(Markdown("### Hidden causal rubric"))
            display(
                JSON(
                    {
                        "mechanism_steps": example["mechanism_steps"],
                        "causal_links": example["causal_links"],
                        "required_concepts": example["required_concepts"],
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 3. Audit causal structure

            This plot does not claim that a longer explanation is better. It checks whether
            the dataset contains explicit multi-step causal supervision rather than isolated
            answer labels.
            """
        ),
        code(
            """
            audit = pd.DataFrame(
                {
                    "split": [row["split"] for row in canonical],
                    "task_chars": [len(row["task"]) for row in canonical],
                    "mechanism_steps": [len(row["mechanism_steps"]) for row in canonical],
                    "causal_links": [len(row["causal_links"]) for row in canonical],
                }
            )
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            sns.histplot(audit, x="mechanism_steps", discrete=True, ax=axes[0], color="#315c8c")
            axes[0].set_title("Explicit steps per causal rubric")
            sns.scatterplot(
                audit,
                x="task_chars",
                y="causal_links",
                hue="split",
                ax=axes[1],
                s=80,
            )
            axes[1].set_title("Task size versus cause→effect links")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 4. Project the SFT view

            SFT receives a conversational `(prompt, completion)` pair. The completion is the
            tagged reference response. Raw source text and the hidden rubric are intentionally
            absent. Paper IDs, URL, and license remain for traceability.
            """
        ),
        code(
            """
            import shutil

            sft = build_sft_dataset(canonical)
            if len(sft["train"]) == 0 or len(sft["validation"]) == 0:
                raise RuntimeError(
                    "The paper-level split produced an empty SFT partition. Increase "
                    "MAX_SOURCE_PAPERS, then rerun generation."
                )
            if SFT_DISK.exists():
                shutil.rmtree(SFT_DISK)
            sft.save_to_disk(SFT_DISK)
            for split_name, split_data in sft.items():
                split_data.to_parquet(DATA / "processed" / f"sft_{split_name}.parquet")
            if PUSH_DATASETS_TO_HUB:
                sft.push_to_hub(
                    DATASET_HF_REPO,
                    config_name="sft",
                    private=DATASET_PRIVATE,
                    token=HF_TOKEN,
                    commit_message="Publish mechanism SFT splits",
                )

            assert "source_text" not in sft["train"].column_names
            assert "mechanism_steps" not in sft["train"].column_names
            print(sft)
            display(JSON(sft["train"][0]))
            """
        ),
        markdown(
            """
            ## Result

            You now have:

            - an auditable canonical dataset with source provenance and hidden causal rubrics;
            - an explicit rejection log;
            - an SFT dataset whose model input is only a self-contained mechanism task.

            The processed SFT splits are also published under the `sft` configuration of
            `lamm-mit/scientific-sft-grpo-data`. The private canonical source-text audit file
            remains local and is never uploaded.

            Continue with **02_build_mechanism_grpo_dataset.ipynb**.
            """
        ),
    ]
)


nb2 = notebook(
    [
        markdown(
            """
            # 02 — Build and inspect the GRPO dataset

            SFT teaches the desired response pattern. GRPO then samples several answers to the
            same task and increases the probability of answers that receive higher reward.

            The student-facing prompt is still only a self-contained mechanism task. Hidden
            reference fields are passed to reward functions, not appended to the prompt.
            """
        ),
        markdown(
            """
            ## Configuration

            The OpenAI key is read from `OPENAI_API_KEY`. All other settings are explicit
            notebook variables. Hugging Face uses your cached `hf auth login` credentials
            unless you deliberately uncomment the optional token line.
            """
        ),
        code(
            """
            import os
            import shutil
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            from IPython.display import JSON, Markdown, display

            from science_course.data import (
                build_grpo_dataset,
                read_jsonl,
                render_completion,
            )
            from science_course.hub import require_hf_namespace
            from science_course.judge import MechanismJudge
            from science_course.rewards import component_scores
            from science_course.teacher import DEFAULT_TEACHER_MODEL, require_openai_key

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            DATA = ROOT / "data"
            CANONICAL = DATA / "canonical" / "mechanism_tasks.jsonl"
            GRPO_DISK = DATA / "processed" / "grpo"
            JUDGE_CACHE = ROOT / "results" / "grpo_judge_cache.jsonl"

            JUDGE_MODEL = DEFAULT_TEACHER_MODEL
            DATASET_HF_REPO = "lamm-mit/scientific-sft-grpo-data"
            PUSH_DATASETS_TO_HUB = True
            DATASET_PRIVATE = False
            HF_TOKEN = None
            # HF_TOKEN = os.environ["HF_TOKEN"]  # Optional; prefer `hf auth login`.

            require_openai_key()
            assert JUDGE_MODEL == "gpt-5.6-terra", (
                "This class notebook is tested with the requested judge model: "
                "gpt-5.6-terra"
            )
            if PUSH_DATASETS_TO_HUB:
                require_hf_namespace(DATASET_HF_REPO, token=HF_TOKEN)
            canonical = read_jsonl(CANONICAL)
            if not canonical:
                raise RuntimeError("Run notebook 01 first.")
            sns.set_theme(style="whitegrid", context="talk")
            print(
                {
                    "canonical_tasks": len(canonical),
                    "judge_model": JUDGE_MODEL,
                    "dataset_hub_repo": DATASET_HF_REPO,
                }
            )
            """
        ),
        markdown(
            """
            ## 1. Make the GRPO projection

            Each row contains:

            - `prompt`: the only input to the policy;
            - `task`: used to verify that quoted evidence actually appears in the task;
            - `mechanism_steps` and `causal_links`: a hidden semantic rubric;
            - reference reasoning/answer: guidance for the judge, not a phrase-match target;
            - provenance fields.
            """
        ),
        code(
            """
            grpo = build_grpo_dataset(canonical)
            if len(grpo["train"]) == 0 or len(grpo["validation"]) == 0:
                raise RuntimeError(
                    "The paper-level split produced an empty GRPO partition. Increase "
                    "MAX_SOURCE_PAPERS in notebook 01, then rerun generation."
                )
            if GRPO_DISK.exists():
                shutil.rmtree(GRPO_DISK)
            grpo.save_to_disk(GRPO_DISK)
            for split_name, split_data in grpo.items():
                split_data.to_parquet(DATA / "processed" / f"grpo_{split_name}.parquet")
            if PUSH_DATASETS_TO_HUB:
                grpo.push_to_hub(
                    DATASET_HF_REPO,
                    config_name="grpo",
                    private=DATASET_PRIVATE,
                    token=HF_TOKEN,
                    commit_message="Publish mechanism GRPO splits",
                )

            for split_data in grpo.values():
                assert "source_text" not in split_data.column_names
                assert "completion" not in split_data.column_names

            print(grpo)
            """
        ),
        code(
            """
            row = grpo["train"][0]
            display(Markdown("### What the policy sees"))
            display(JSON(row["prompt"]))
            display(Markdown("### What reward functions can see"))
            display(
                JSON(
                    {
                        key: row[key]
                        for key in (
                            "task",
                            "mechanism_steps",
                            "causal_links",
                            "required_concepts",
                            "reference_answer",
                        )
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 2. Reward design: mechanics plus semantics

            Four complementary signals are used:

            1. **Structure** — the three required response tags are present.
            2. **Evidence grounding** — the quoted evidence occurs in the task.
            3. **Concept coverage** — expected concepts appear somewhere in the explanation.
            4. **Mechanism judge** — `gpt-5.6-terra` evaluates causal correctness,
               completeness, and whether evidence supports the stated mechanism.

            The semantic judge receives a whole trainer batch in one structured-output API
            request. Judgments are cached by content hash, including model and prompt version.
            A keyword check cannot tell causal direction; the judge carries most of the reward.
            """
        ),
        code(
            """
            reference_completion = render_completion(
                {
                    "reasoning": row["reference_reasoning"],
                    "evidence": row["reference_evidence"],
                    "answer": row["reference_answer"],
                }
            )
            deterministic = component_scores(
                reference_completion,
                task=row["task"],
                reference_answer=row["reference_answer"],
                required_concepts=row["required_concepts"],
            )
            display(JSON(deterministic))
            """
        ),
        code(
            """
            judge = MechanismJudge(
                model=JUDGE_MODEL,
                cache_path=JUDGE_CACHE,
            )
            semantic_score = judge.score(
                [
                    {
                        "task": row["task"],
                        "reference_reasoning": row["reference_reasoning"],
                        "reference_answer": row["reference_answer"],
                        "mechanism_steps": row["mechanism_steps"],
                        "causal_links": row["causal_links"],
                        "student_response": reference_completion,
                    }
                ]
            )[0]
            print({"cached_semantic_mechanism_score": semantic_score})
            """
        ),
        markdown(
            """
            ## 3. Inspect the mechanism curriculum

            The purpose of this audit is diversity of causal structure, not numerical label
            balance. We inspect task length, number of causal links, and concept vocabulary.
            """
        ),
        code(
            """
            frame = pd.DataFrame(
                [
                    {
                        "split": split_name,
                        "task_chars": len(item["task"]),
                        "causal_links": len(item["causal_links"]),
                        "mechanism_steps": len(item["mechanism_steps"]),
                    }
                    for split_name, split_data in grpo.items()
                    for item in split_data
                ]
            )
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            sns.boxplot(frame, x="split", y="task_chars", ax=axes[0], color="#87a9cc")
            axes[0].tick_params(axis="x", rotation=25)
            axes[0].set_title("Self-contained task size")
            sns.countplot(
                frame,
                x="causal_links",
                hue="split",
                ax=axes[1],
                palette="deep",
            )
            axes[1].set_title("Explicit causal links")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            concept_counts = (
                pd.Series(
                    [
                        concept.casefold()
                        for split_data in grpo.values()
                        for item in split_data
                        for concept in item["required_concepts"]
                    ]
                )
                .value_counts()
                .head(20)
                .sort_values()
            )
            concept_counts.plot.barh(
                figsize=(9, 7),
                color="#d97732",
                title="Frequent rubric concepts",
            )
            plt.xlabel("tasks")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## Result

            The GRPO policy will receive no answer and no source document—only a
            self-contained mechanism task. The hidden causal rubric supports semantic reward.

            Continue with **03_finetune_sft_lora.ipynb**.
            """
        ),
    ]
)


nb3 = notebook(
    [
        markdown(
            """
            # 03 — LoRA supervised fine-tuning (SFT)

            We adapt the instruction-tuned Gemma 4 text decoder with LoRA. The base weights
            remain frozen; small low-rank adapters learn to produce:

            ```
            <reasoning>causal chain</reasoning>
            <evidence>observation quoted from the task</evidence>
            <answer>mechanistic conclusion</answer>
            ```

            Device selection is automatic: **CUDA → MPS → CPU**. Gemma 4 E4B is a serious
            model; MPS and CPU paths are functional fallbacks but may be impractically slow
            on a laptop. Set `MODEL_ID` to a smaller chat model for an in-class laptop run.
            """
        ),
        markdown(
            """
            ## Prerequisites

            Run notebooks 01–02 first. Accept the Gemma license on Hugging Face and authenticate:

            ```bash
            hf auth login
            ```

            Every model, LoRA, training, checkpoint, and Hub setting is defined in the
            configuration cell below.
            """
        ),
        markdown(
            """
            ## Configuration

            Change values here—do not create additional environment variables. Hugging Face
            uses cached login credentials. The optional `HF_TOKEN` line may be uncommented
            after exporting a token, but never paste a token into the notebook.
            """
        ),
        code(
            """
            import os

            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from datasets import load_from_disk
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

            # Runtime and artifact locations
            ENABLE_MPS_FALLBACK = True
            if ENABLE_MPS_FALLBACK:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            MODEL_ID = "google/gemma-4-E4B-it"
            SFT_DATA = ROOT / "data" / "processed" / "sft"
            OUTPUT_DIR = ROOT / "artifacts" / "gemma4-mechanism-sft"
            RESUME_FROM_CHECKPOINT = None

            # LoRA
            LORA_RANK = 16
            LORA_ALPHA = 32
            LORA_DROPOUT = 0.05
            LORA_TARGET_MODULES = "all-linear"

            # SFT
            NUM_TRAIN_EPOCHS = 3
            MAX_STEPS = -1
            LEARNING_RATE = 1e-4
            TRAIN_BATCH_SIZE = 1
            EVAL_BATCH_SIZE = 1
            GRADIENT_ACCUMULATION_STEPS = 8
            MAX_SEQUENCE_LENGTH = 1_024
            WARMUP_RATIO = 0.05
            EVAL_STEPS = 25
            SAVE_STEPS = 25
            LOGGING_STEPS = 5
            RANDOM_SEED = 17

            # Hugging Face publication: every saved checkpoint plus the final adapter
            DATASET_HF_REPO = "lamm-mit/scientific-sft-grpo-data"
            SFT_HF_REPO = "lamm-mit/scientific-sft-grpo-sft"
            PUSH_TO_HUB = True
            HUB_PRIVATE_REPO = False
            HF_TOKEN = None
            # HF_TOKEN = os.environ["HF_TOKEN"]  # Optional; prefer `hf auth login`.

            versions = require_training_stack()
            runtime = detect_runtime()
            if PUSH_TO_HUB:
                require_hf_namespace(SFT_HF_REPO, token=HF_TOKEN)

            display(
                JSON(
                    {
                        "runtime": runtime.as_dict(),
                        "versions": versions,
                        "model": MODEL_ID,
                        "hub_repo": SFT_HF_REPO,
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 1. Load and render the prompt-completion data

            The tokenizer's own chat template formats the prompt. We disable any model-specific
            thinking mode when the template supports that option. SFT loss is computed on the
            reference completion, not the prompt.
            """
        ),
        code(
            """
            if not SFT_DATA.exists():
                raise RuntimeError("Run notebook 01 to create the SFT dataset.")
            sft = load_from_disk(SFT_DATA)
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
                    split_name: pd.Series([
                        len(tokenizer(item["prompt"] + item["completion"]).input_ids)
                        for item in split_data
                    ])
                    for split_name, split_data in rendered.items()
                }
            )
            lengths.plot.hist(bins=20, alpha=0.65, figsize=(9, 4.5))
            plt.axvline(
                MAX_SEQUENCE_LENGTH,
                color="crimson",
                linestyle="--",
                label="training max_length",
            )
            plt.xlabel("tokens")
            plt.title("SFT sequence-length audit")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 2. Load Gemma and attach LoRA

            `AutoModelForCausalLM` loads only the text decoder. We do not quantize because the
            same notebook must work on CUDA, Apple MPS, and CPU. LoRA targets linear layers;
            only adapter parameters are optimized.
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
                completion_only_loss=True,
                loss_type="nll",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                optim="adamw_torch",
                warmup_ratio=WARMUP_RATIO,
                eval_strategy="steps",
                eval_steps=EVAL_STEPS,
                save_strategy="steps",
                save_steps=SAVE_STEPS,
                save_total_limit=None,
                logging_steps=LOGGING_STEPS,
                logging_first_step=True,
                report_to="none",
                push_to_hub=PUSH_TO_HUB,
                hub_model_id=SFT_HF_REPO,
                hub_strategy="all_checkpoints",
                hub_private_repo=HUB_PRIVATE_REPO,
                hub_token=HF_TOKEN,
                hub_always_push=True,
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
            ## 3. Train

            The progress bar reports optimization loss; periodic validation measures whether
            held-out reference responses are also becoming more likely. Change `MAX_STEPS`
            in the configuration cell for a bounded classroom run.
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
                    commit_message="Complete mechanism SFT training"
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
                    x="step", y="eval_loss", ax=axes[1], color="#d97732", legend=False
                )
            axes[1].set_title("Held-out SFT loss")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 4. Use the adapter on a new task

            This is the real deployment interface: give the fine-tuned model a new
            self-contained mechanism task. No original paper text or reference answer is
            required at inference time.
            """
        ),
        code(
            """
            new_task = (
                "A hydrogel contains polymer chains joined by reversible host–guest "
                "interactions. Mechanical strain separates some pairs, allowing local chain "
                "rearrangement. After the strain is removed, compatible host and guest groups "
                "associate again. Explain how these events allow the gel to recover after damage."
            )
            new_messages = [
                {
                    "role": "system",
                    "content": (
                        "Solve the self-contained scientific mechanism task. Explain the causal "
                        "chain, quote the most relevant observation already included in the task, "
                        "and give a concise answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"SCIENTIFIC MECHANISM TASK\\n{new_task}\\n\\n"
                        "Respond with <reasoning>, <evidence>, and <answer> tags."
                    ),
                },
            ]
            prompt = render_prompt(tokenizer, new_messages)
            response = generate_text(trainer.model, tokenizer, prompt, runtime)
            display(Markdown(f"```text\\n{response}\\n```"))
            """
        ),
        markdown(
            """
            ## Result

            `artifacts/gemma4-mechanism-sft/` contains a small LoRA adapter plus tokenizer
            metadata—not a second copy of all Gemma weights. The final adapter and every saved
            checkpoint are also published to `lamm-mit/scientific-sft-grpo-sft`. Notebook 04
            loads that adapter as the starting policy and improves it with mechanism-sensitive
            GRPO rewards.
            """
        ),
    ]
)


nb4 = notebook(
    [
        markdown(
            """
            # 04 — Continue the LoRA adapter with GRPO

            GRPO samples a group of candidate explanations for each self-contained task,
            scores them, and reinforces answers that are better **relative to the group**.

            ```
                                  ┌─ candidate A ─ rewards ─┐
            mechanism task ──────┼─ candidate B ─ rewards ─┼─ relative advantages ─ LoRA update
                                  ├─ candidate C ─ rewards ─┤
                                  └─ candidate D ─ rewards ─┘
            ```

            This notebook continues the SFT LoRA adapter. It does not restart from the base
            model, and it never trains against numerical target values.
            """
        ),
        markdown(
            """
            ## Important execution note

            The semantic reward calls `gpt-5.6-terra`. Use a single training process in this
            teaching notebook; the batched judge cache is intentionally simple and auditable.
            Each unique completion is paid API work, so begin with a short run and inspect the
            logged rewards before increasing `MAX_STEPS` in the configuration cell.
            """
        ),
        markdown(
            """
            ## Configuration

            All models, rewards, training hyperparameters, paths, and Hub repositories are
            explicit here. `OPENAI_API_KEY` remains an environment secret. Hugging Face uses
            `hf auth login`; the optional token line may be uncommented but never populated
            with a literal token inside the notebook.
            """
        ),
        code(
            """
            import os

            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from accelerate import PartialState
            from datasets import load_from_disk
            from IPython.display import JSON, Markdown, display
            from peft import PeftModel
            from trl import GRPOConfig, GRPOTrainer

            from science_course.devices import clear_device_cache, detect_runtime
            from science_course.hub import require_hf_namespace
            from science_course.judge import (
                configure_mechanism_judge,
                mechanism_judge_reward,
            )
            from science_course.modeling import (
                generate_text,
                load_causal_lm,
                load_tokenizer,
                render_prompt,
            )
            from science_course.rewards import (
                concept_coverage_reward,
                evidence_grounding_reward,
                format_reward,
            )
            from science_course.teacher import DEFAULT_TEACHER_MODEL, require_openai_key
            from science_course.versions import require_training_stack

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent

            # Runtime and artifacts
            ENABLE_MPS_FALLBACK = True
            if ENABLE_MPS_FALLBACK:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            MODEL_ID = "google/gemma-4-E4B-it"
            SFT_ADAPTER_SOURCE = ROOT / "artifacts" / "gemma4-mechanism-sft"
            GRPO_ADAPTER = ROOT / "artifacts" / "gemma4-mechanism-grpo"
            GRPO_DATA = ROOT / "data" / "processed" / "grpo"
            RESUME_FROM_CHECKPOINT = None

            # Semantic judge
            JUDGE_MODEL = DEFAULT_TEACHER_MODEL
            JUDGE_CACHE = ROOT / "results" / "grpo_judge_cache.jsonl"

            # GRPO
            MAX_STEPS = 25
            LEARNING_RATE = 5e-6
            TRAIN_BATCH_SIZE = 1
            EVAL_BATCH_SIZE = 1
            GRADIENT_ACCUMULATION_STEPS = 4
            NUM_GENERATIONS = 4
            NUM_GENERATIONS_EVAL = 4
            MAX_COMPLETION_LENGTH = 256
            TEMPERATURE = 0.8
            TOP_P = 0.95
            BETA = 0.0
            REWARD_WEIGHTS = [0.10, 0.15, 0.15, 0.60]
            WARMUP_RATIO = 0.05
            EVAL_STEPS = 10
            SAVE_STEPS = 10
            LOGGING_STEPS = 1
            NUM_COMPLETIONS_TO_PRINT = 4
            RANDOM_SEED = 17

            # Hugging Face publication: every GRPO checkpoint plus the final adapter
            DATASET_HF_REPO = "lamm-mit/scientific-sft-grpo-data"
            SFT_HF_REPO = "lamm-mit/scientific-sft-grpo-sft"
            GRPO_HF_REPO = "lamm-mit/scientific-sft-grpo-grpo"
            PUSH_TO_HUB = True
            HUB_PRIVATE_REPO = False
            HF_TOKEN = None
            # HF_TOKEN = os.environ["HF_TOKEN"]  # Optional; prefer `hf auth login`.

            require_openai_key()
            versions = require_training_stack()
            runtime = detect_runtime()
            if JUDGE_MODEL != "gpt-5.6-terra":
                raise RuntimeError("The tested semantic judge is gpt-5.6-terra.")
            if PartialState().num_processes != 1:
                raise RuntimeError("This API-judged teaching run requires WORLD_SIZE=1.")
            configure_mechanism_judge(
                model=JUDGE_MODEL,
                cache_path=JUDGE_CACHE,
            )
            if PUSH_TO_HUB:
                require_hf_namespace(GRPO_HF_REPO, token=HF_TOKEN)

            display(
                JSON(
                    {
                        "runtime": runtime.as_dict(),
                        "versions": versions,
                        "model": MODEL_ID,
                        "sft_adapter": str(SFT_ADAPTER_SOURCE),
                        "grpo_hub_repo": GRPO_HF_REPO,
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## 1. Load the GRPO data and the trainable SFT adapter

            Hidden columns remain in the dataset because custom reward functions receive them.
            The model receives only `prompt`. Loading the SFT adapter with `is_trainable=True`
            ensures GRPO updates the same LoRA parameters.
            """
        ),
        code(
            """
            if not GRPO_DATA.exists():
                raise RuntimeError("Run notebook 02 to create the GRPO dataset.")
            if isinstance(SFT_ADAPTER_SOURCE, Path) and not SFT_ADAPTER_SOURCE.exists():
                raise RuntimeError("Run notebook 03 to create the SFT adapter.")

            grpo = load_from_disk(GRPO_DATA)
            if len(grpo["train"]) == 0:
                raise RuntimeError("The GRPO train split is empty.")

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
            ## 2. Establish a pre-GRPO behavior snapshot

            We keep the same held-out prompt before and after training. This is a qualitative
            teaching comparison, not a claim of scientific benchmark performance.
            """
        ),
        code(
            """
            held_out_pool = grpo["test"] if len(grpo["test"]) else grpo["validation"]
            if not len(held_out_pool):
                raise RuntimeError("A validation or test task is required for comparison.")
            held_out = held_out_pool[0]
            held_out_prompt = render_prompt(tokenizer, held_out["prompt"])
            before_grpo = generate_text(model, tokenizer, held_out_prompt, runtime)
            display(Markdown("### SFT policy\\n```text\\n" + before_grpo + "\\n```"))
            """
        ),
        markdown(
            """
            ## 3. Configure mechanism-sensitive GRPO

            Reward weights intentionally make causal judgment dominant:

            - format: 0.10
            - task-grounded evidence: 0.15
            - rubric concept coverage: 0.15
            - batched `gpt-5.6-terra` mechanism judge: 0.60

            Deterministic checks constrain obvious failure modes; the semantic judge evaluates
            causal direction, intermediate steps, support, and scientific coherence.
            """
        ),
        code(
            """
            grpo_args = GRPOConfig(
                output_dir=str(GRPO_ADAPTER),
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
                beta=BETA,
                reward_weights=REWARD_WEIGHTS,
                scale_rewards="group",
                loss_type="dapo",
                remove_unused_columns=False,
                chat_template_kwargs={"enable_thinking": False},
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                optim="adamw_torch",
                warmup_ratio=WARMUP_RATIO,
                eval_strategy="steps",
                eval_steps=EVAL_STEPS,
                save_strategy="steps",
                save_steps=SAVE_STEPS,
                save_total_limit=None,
                logging_steps=LOGGING_STEPS,
                logging_first_step=True,
                log_completions=True,
                num_completions_to_print=NUM_COMPLETIONS_TO_PRINT,
                report_to="none",
                push_to_hub=PUSH_TO_HUB,
                hub_model_id=GRPO_HF_REPO,
                hub_strategy="all_checkpoints",
                hub_private_repo=HUB_PRIVATE_REPO,
                hub_token=HF_TOKEN,
                hub_always_push=True,
                bf16=runtime.trainer_bf16,
                fp16=runtime.trainer_fp16,
                use_cpu=runtime.use_cpu,
                use_vllm=False,
                seed=RANDOM_SEED,
            )

            trainer = GRPOTrainer(
                model=model,
                args=grpo_args,
                reward_funcs=[
                    format_reward,
                    evidence_grounding_reward,
                    concept_coverage_reward,
                    mechanism_judge_reward,
                ],
                train_dataset=grpo["train"],
                eval_dataset=grpo["validation"],
                processing_class=tokenizer,
            )
            """
        ),
        markdown(
            """
            ## 4. Train and inspect reward dynamics

            A rising total reward is not enough: inspect the individual components. If format
            rises while semantic reward stagnates, the policy is learning the shell but not a
            better mechanism.
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
                    commit_message="Complete mechanism GRPO training"
                )
                print(f"Published final adapter and all checkpoints: {hub_result}")
            display(JSON(train_result.metrics))
            """
        ),
        code(
            """
            history = pd.DataFrame(trainer.state.log_history)
            reward_columns = [
                column
                for column in history.columns
                if column == "reward"
                or (column.startswith("rewards/") and column.endswith("/mean"))
            ]
            if reward_columns:
                history[["step", *reward_columns]].dropna(how="all", subset=reward_columns).plot(
                    x="step", y=reward_columns, figsize=(12, 5), marker="o"
                )
                plt.title("GRPO reward components")
                plt.ylabel("reward")
                plt.tight_layout()
                plt.show()
            else:
                print("Reward columns available after this run:", sorted(history.columns))
            """
        ),
        markdown(
            """
            ## 5. Compare the same unseen mechanism task

            Look for an ordered, supported causal explanation—not a particular phrase. The
            source document is unnecessary because the task itself contains the relevant
            observations.
            """
        ),
        code(
            """
            after_grpo = generate_text(trainer.model, tokenizer, held_out_prompt, runtime)
            display(Markdown("### Before GRPO\\n```text\\n" + before_grpo + "\\n```"))
            display(Markdown("### After GRPO\\n```text\\n" + after_grpo + "\\n```"))
            display(Markdown("### Hidden reference (for instructor inspection)"))
            display(
                JSON(
                    {
                        "mechanism_steps": held_out["mechanism_steps"],
                        "causal_links": held_out["causal_links"],
                        "reference_answer": held_out["reference_answer"],
                    }
                )
            )
            """
        ),
        markdown(
            """
            ## Result and responsible interpretation

            `artifacts/gemma4-mechanism-grpo/` is the final LoRA adapter. The final adapter and
            every saved checkpoint are also published to
            `lamm-mit/scientific-sft-grpo-grpo`. At inference, attach it to the same base model
            and submit any self-contained scientific mechanism task.

            For real research, add expert adjudication, inter-rater reliability, domain-specific
            test sets, ablations for each reward component, contamination checks, and multiple
            seeds. An LLM judge is scalable supervision—not ground truth.
            """
        ),
    ]
)


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "01_generate_mechanism_sft_dataset.ipynb": nb1,
        "02_build_mechanism_grpo_dataset.ipynb": nb2,
        "03_finetune_sft_lora.ipynb": nb3,
        "04_finetune_grpo_lora.ipynb": nb4,
    }
    for name, artifact in outputs.items():
        path = NOTEBOOKS / name
        nbf.write(artifact, path)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
