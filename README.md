# Scientific Problem-Solving SFT + GRPO

Four teaching notebooks plus reproducible dataset-generation and training CLIs that turn
openly licensed scientific text into task-only problem-solving datasets, train a Gemma 4
LoRA adapter with supervised fine-tuning, and continue it with a simple `gpt-5.6-luna` GRPO
reward.

The student model never receives a paper passage. Source text is used once by a teacher and
critic to author a new self-contained task, then removed from the policy-facing datasets.

## What the model learns

Every task requests the same inspectable scientific work product:

```text
<brainstorm>
three to five distinct candidate ideas
</brainstorm>
<principles>
scientific constraints and design principles
</principles>
<synthesis>
compare or combine the candidates using the principles
</synthesis>
<answer>
a concise final proposal or conclusion
</answer>
```

The tasks span:

- mechanism-guided design;
- experimental design;
- troubleshooting and failure analysis;
- hypothesis development; and
- cross-domain synthesis.

They are qualitative and self-contained. They do not ask the student to retrieve a paper,
quote a passage, or calculate a numerical target.

## Pipeline

```text
open scientific source
        │
        ├─ gpt-5.6-terra teacher: creates a new task, reference, and rubric
        └─ gpt-5.6-terra critic: checks the record against the source
                         │
                         ▼
              accepted canonical record
                 ├─ SFT: task + structured reference completion
                 └─ GRPO: task + hidden grading rubric
                                      │
                                      ▼
                         Gemma 4 LoRA: SFT → GRPO
                                      │
                                      ▼
                              gpt-5.6-luna reward
```

SFT and GRPO use separate source-paper pools. GRPO train, validation, and final-test splits
are also explicit and disjoint.

## Default dataset scale

Generation is based on accepted-example quotas, not a fixed number of teacher attempts:

| Partition | Accepted examples |
|---|---:|
| SFT train | 500 |
| SFT validation | 75 |
| GRPO train | 500 |
| GRPO validation | 75 |
| Final test | 100 |
| Total | 1,250 |

Within each split, exact integer quotas follow the task-family weights declared in the
notebook. Rejected drafts do not consume a quota. The generator continues until every quota is
full or the visible source/attempt budget is exhausted.

Generation is concurrent, incrementally persisted, and safe to resume. With an acceptance
rate near one third, the defaults may require several thousand source candidates and
substantial API usage.

## Exact GRPO reward

The reward intentionally has only two conceptual components:

1. deterministic format validation; and
2. one semantic judgment from `gpt-5.6-luna`.

For every completion:

\[
R = F \times (0.10 + 0.90J)
\]

- `F=1` only when all four non-empty sections appear in the correct order, with no text
  outside the tags; otherwise `F=0`.
- Luna assigns integer scores from 0–4 for `brainstorm`, `principles`, `synthesis`, and
  `answer`.
- `J=(B+P+S+A)/16`.
- Malformed completions receive exactly zero reward and do not trigger a judge API call.

The judge receives the task, required constraints, evaluation criteria, acceptable
alternatives, failure modes, and candidate response. It does not compare against a single
gold answer. Valid scientific alternatives are explicitly allowed.

All candidates in a trainer batch are judged in one structured-output request. Results are
cached using a hash of the task, rubric, completion, judge model, and judge-prompt version.
TRL then normalizes raw rewards within each four-completion group.

The default GRPO run now performs held-out evaluation once per epoch
(`eval_strategy = "epoch"`), while checkpoints are still saved every 25 steps. One complete
evaluation generates four rollouts for each of 75 validation tasks—up to 300 Luna-graded
responses—so evaluating every 25 training steps is unnecessarily expensive for this class
exercise. Set `eval_strategy = "steps"` and `eval_steps = 250` in the notebook or TOML if
periodic validation is preferred.

The API judge is needed only during GRPO training. The final adapter does not call Luna at
inference.

## Installation

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/lamm-mit/scientific-sft-grpo.git
cd scientific-sft-grpo

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Accept the Gemma license and authenticate:

```bash
hf auth login
export OPENAI_API_KEY="..."
jupyter lab
```

There is no offline synthetic-data mode. Only `OPENAI_API_KEY` is required from the
environment. Hugging Face uses cached login credentials; every notebook also contains an
optional commented `HF_TOKEN = os.environ["HF_TOKEN"]` line.

## Notebook order

| Notebook | Purpose | Main artifact |
|---|---|---|
| `01_generate_scientific_design_sft_dataset.ipynb` | Generate 500/75 accepted SFT tasks, audit, project, and publish | `data/processed/scientific_design_sft/` |
| `02_generate_scientific_design_grpo_dataset.ipynb` | Generate paper-disjoint 500/75/100 GRPO tasks, demonstrate Luna reward, and publish | `data/processed/scientific_design_grpo/` |
| `03_finetune_sft_lora.ipynb` | Train completion-only Gemma 4 LoRA SFT | `artifacts/gemma4-scientific-design-sft/` |
| `04_finetune_grpo_lora.ipynb` | Continue the SFT adapter with grouped Luna-judged GRPO | `artifacts/gemma4-scientific-design-grpo/` |

Notebooks 03 and 04 are Hub-first: a fresh clone downloads the published datasets and
the SFT adapter directly from the repositories below. You do not need to run notebooks 01–03
before starting notebook 04. To use locally generated artifacts instead, change the explicit
`*_SOURCE_MODE` variables from `"hub"` to `"local"` in the corresponding training notebook.

Notebook 04 ends with a fresh-kernel inference cell. It independently loads the base model and
final GRPO adapter from Hugging Face, so it still works after closing and reopening Jupyter.
The same cell can load a specific published `checkpoint-*` subfolder, revision, tag, or commit.
Inference does not use Luna and does not require `OPENAI_API_KEY`.

## Dataset generation from the CLI

Notebooks 01 and 02 remain the visual teaching path. The two additional CLI commands run the
same accepted-quota teacher/critic workflow without Jupyter:

```bash
scientific-sft-grpo generate-sft --config configs/generate_sft.toml
scientific-sft-grpo generate-grpo --config configs/generate_grpo.toml
```

The SFT command:

1. streams and license-filters `common-pile/peS2o`;
2. asks `gpt-5.6-terra` to author a self-contained task, reference work product, and hidden
   rubric;
3. uses an independent Terra critic call to accept or reject it;
4. continues until every SFT split/family accepted quota is full;
5. writes canonical accepted/rejected JSONL journals;
6. projects only task + reference completion into SFT Arrow and Parquet outputs; and
7. optionally publishes the configured Hugging Face dataset configuration.

The GRPO command follows the same process, but first reads the SFT canonical journal and
excludes every SFT paper from its source pool. Its policy-facing projection contains the task
and hidden rubric, but no reference completion. GRPO generation must therefore run after the
matching SFT generation.

Both commands are automatically resumable. Every completed source is immediately appended to
the accepted or rejected journal. Rerunning the identical command skips those source papers
and continues the remaining quotas. A manifest prevents a resume from silently mixing
different teacher models, task-family weights, source filters, seeds, or prompt versions.
Increasing `candidate_limit` and `max_attempts` is allowed if the initial budget is
insufficient; decreasing accepted targets or changing the immutable contract requires a new
output name and paths.

Inspect a generation job without making API calls:

```bash
scientific-sft-grpo show-config \
  --stage generate-sft \
  --config configs/generate_sft.toml

scientific-sft-grpo show-config \
  --stage generate-grpo \
  --config configs/generate_grpo.toml
```

Use `--no-push` to produce only local Arrow and Parquet artifacts:

```bash
scientific-sft-grpo generate-sft \
  --config configs/generate_sft.toml \
  --no-push
```

### Large `L` datasets: 10,000 SFT and 1,200 GRPO

Two isolated large-run configurations are committed. They do not overwrite the teaching
datasets:

| Configuration | Splits | Accepted total | Hub configuration |
|---|---|---:|---|
| `generate_sft_L.toml` | 9,000 train + 1,000 validation | 10,000 | `scientific_design_sft_L` |
| `generate_grpo_L.toml` | 1,000 train + 100 validation + 100 test | 1,200 | `scientific_design_grpo_L` |

Run them in order:

```bash
export OPENAI_API_KEY="..."
hf auth login

mkdir -p logs

scientific-sft-grpo generate-sft \
  --config configs/generate_sft_L.toml \
  2>&1 | tee logs/generate-sft-L.log

scientific-sft-grpo generate-grpo \
  --config configs/generate_grpo_L.toml \
  2>&1 | tee logs/generate-grpo-L.log
```

The large SFT configuration allows up to 40,000 source attempts and the GRPO configuration
allows up to 5,000. An attempt always makes a teacher call; teacher-valid drafts also make a
critic call. These are maximum budgets rather than cost estimates. Review your OpenAI rate
limits and expected spend before starting. Concurrency is set to 16 and can be reduced in TOML
without invalidating a resume.

For a long remote run, put the same command in `tmux`. Progress reports accepted examples,
not merely attempted sources. If generation stops with quota deficits, increase
`source.candidate_limit`, `source.max_records_scanned`, and
`generation.max_attempts`, then rerun the same command.

The large GRPO config points to
`data/canonical/scientific_design_sft_L_tasks.jsonl`, guaranteeing its source pool excludes the
10,000-example SFT curriculum. Both `L` datasets publish as separate configurations inside
`lamm-mit/scientific-sft-grpo-data`.

## Training from the CLI

The notebooks remain the visual, explanatory teaching path. The CLI is an additional
non-interactive path for a DGX, workstation, remote VS Code terminal, `tmux`, or a scheduler.
It uses the same datasets, model, LoRA settings, reward, and training defaults.

First inspect the fully parsed settings:

```bash
scientific-sft-grpo show-config --stage sft --config configs/sft.toml
scientific-sft-grpo show-config --stage grpo --config configs/grpo.toml
```

Run the read-only preflight. It checks the installed stack, selected accelerator, gated model
access, dataset schema, checkpoint state, and Hub write namespace. GRPO also checks the SFT
adapter, `OPENAI_API_KEY`, and the required single-process execution mode.

```bash
scientific-sft-grpo doctor --stage sft --config configs/sft.toml
scientific-sft-grpo doctor --stage grpo --config configs/grpo.toml
```

Train SFT and then GRPO:

```bash
scientific-sft-grpo sft --config configs/sft.toml
scientific-sft-grpo grpo --config configs/grpo.toml
```

Both committed configurations use `resume = "auto"`. The CLI resumes the numerically latest
local `checkpoint-*` directory when one exists and starts at step zero otherwise. It never
silently downloads a Hub checkpoint as trainer state. Override the behavior explicitly:

```bash
scientific-sft-grpo sft --config configs/sft.toml --resume none
scientific-sft-grpo grpo --config configs/grpo.toml --resume auto
scientific-sft-grpo grpo \
  --config configs/grpo.toml \
  --resume artifacts/gemma4-scientific-design-grpo/checkpoint-100
```

For a one-step installation and memory check:

```bash
scientific-sft-grpo sft --config configs/sft.toml --smoke-test
scientific-sft-grpo grpo --config configs/grpo.toml --smoke-test
```

A smoke test selects at most eight training and four validation examples, writes to a
separate `*-smoke` output directory, ignores existing checkpoints, and disables all Hub
writes. The GRPO smoke test still creates real completions and real Luna judge calls.

Normal runs follow the `push = true` setting and publish every trainer checkpoint plus the
final adapter. Use `--no-push` for a local run or `--push` to override a disabled config.
Every output directory also contains:

- `resolved_config.json`: secret-free configuration, package versions, device, and resume
  decision;
- `train_metrics.json`: final `trainer.train()` metrics; and
- `log_history.jsonl`: the complete trainer history for later plots and analysis.

Edit [`configs/sft.toml`](configs/sft.toml) or
[`configs/grpo.toml`](configs/grpo.toml) to change the full training run. Tokens are
intentionally not accepted in TOML: use cached `hf auth login`, optional `HF_TOKEN`, and
`OPENAI_API_KEY`.

## Dataset structure

The local canonical records retain:

- source provenance and source text;
- teacher and critic response IDs and token usage;
- task family and student-visible task;
- reference brainstorm, principles, synthesis, and answer;
- required constraints and evaluation criteria;
- acceptable alternative solutions and failure modes; and
- generation prompt version.

The policy-facing projections differ:

| Field | Canonical | SFT | GRPO |
|---|:---:|:---:|:---:|
| task-only prompt | ✓ | ✓ | ✓ |
| reference structured completion | ✓ | ✓ | — |
| hidden task-specific rubric | ✓ | — | reward only |
| raw source text | ✓ | — | — |
| source license and URL | ✓ | ✓ | ✓ |

TRL receives the GRPO rubric columns for the reward function, but the policy receives only
`prompt`.

## Explicit parameters

Every non-secret setting used by the workflow appears in a notebook configuration cell. The
training settings are mirrored in the two committed TOML files for CLI execution, including:

- accepted split targets and task-family weights;
- source pool and scan limits;
- teacher, critic, and judge models;
- generation concurrency and attempt budget;
- local paths and Hugging Face repositories/config names;
- reward coefficients;
- base model and inference lengths;
- LoRA rank, scaling, dropout, and target modules;
- SFT/GRPO optimization and sampling settings;
- evaluation, logging, checkpoint, and resume settings; and
- Hub publication behavior.

Change notebook variables rather than creating additional environment variables.

## Hardware

`science_course.devices.detect_runtime()` selects:

1. CUDA, preferring BF16 when supported;
2. Apple MPS with FP16; or
3. CPU with FP32.

No CUDA-only quantization package is required. Gemma 4 E4B LoRA training is most practical on
CUDA, but the same code path runs on MPS and CPU.

## Hugging Face publication

Publication defaults:

| Repository/configuration | Contents |
|---|---|
| `lamm-mit/scientific-sft-grpo-data` / `scientific_design_sft` | Public SFT train and validation splits |
| `lamm-mit/scientific-sft-grpo-data` / `scientific_design_grpo` | Public GRPO train, validation, and test splits |
| `lamm-mit/scientific-sft-grpo-design-sft` | Final SFT adapter and every saved checkpoint |
| `lamm-mit/scientific-sft-grpo-design-grpo` | Final GRPO adapter and every saved checkpoint |

The default training inputs are:

- notebook 03: `lamm-mit/scientific-sft-grpo-data`, configuration
  `scientific_design_sft`;
- notebook 04: `lamm-mit/scientific-sft-grpo-data`, configuration
  `scientific_design_grpo`, plus the
  `lamm-mit/scientific-sft-grpo-design-sft` adapter.

Hugging Face caches these downloads. Gemma itself remains gated, so accept Google's Gemma
license and run `hf auth login` before training.

Trainer publication uses `hub_strategy="all_checkpoints"` plus an explicit final push.

## Tested library line

The project pins the releases checked on 2026-07-19:

- Transformers 5.14.1
- TRL 1.8.0
- PEFT 0.19.1
- Datasets 5.0.0
- OpenAI Python 2.46.0

The notebooks use the OpenAI Responses API with structured Pydantic outputs.

## Source licensing

The default source is
[`common-pile/peS2o`](https://huggingface.co/datasets/common-pile/peS2o). The loader filters
license metadata and preserves per-record attribution. Inspect provenance and applicable
licenses before publishing a derived dataset. The Apache-2.0 repository license covers this
code, not the source documents.

## Development checks

```bash
python scripts/build_notebooks.py
python scripts/validate_notebooks.py
PYTHONPATH=src pytest
PYTHONPATH=src ruff check src tests scripts
```

Generated source data, API caches, executed notebooks, adapters, and checkpoints are
gitignored.

## Research caveat

These notebooks teach a complete modern workflow. Scientific claims require expert grading,
inter-rater agreement, multiple seeds, reward ablations, contamination analysis, and
domain-specific held-out benchmarks. An LLM judge is scalable supervision—not ground truth.

## License

Code and notebooks are licensed under the [Apache License 2.0](LICENSE). Source documents and
derived records retain their own attribution and licensing requirements.
