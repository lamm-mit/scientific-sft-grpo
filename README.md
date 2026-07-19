# Scientific Problem-Solving SFT + GRPO

Four teaching notebooks that turn openly licensed scientific text into task-only
problem-solving datasets, train a Gemma 4 LoRA adapter with supervised fine-tuning, and
continue it with a simple `gpt-5.6-luna` GRPO reward.

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

## Parameters are notebook-local

Every non-secret setting used by the workflow appears in a notebook configuration cell,
including:

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
