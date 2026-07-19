# Scientific Mechanism SFT + GRPO Course

Four notebooks that turn openly licensed scientific text into qualitative
mechanism tasks, then adapt an instruction-tuned language model with LoRA:

1. generate and critique a supervised fine-tuning dataset;
2. build a separate GRPO dataset with hidden causal rubrics;
3. train a LoRA adapter with SFT;
4. continue the same adapter with mechanism-sensitive GRPO.

The student model does **not** receive a paper excerpt. Source text is used once by
`gpt-5.6-terra` to author and verify a self-contained task, then removed from both training
views. The target skill is explaining **how or why a mechanism works**, not calculating or
predicting numerical values.

## What the model learns

A training item asks a qualitative question such as:

> A material contains chains joined by reversible interactions. A perturbation separates
> some interactions and permits chain rearrangement; removing the perturbation allows the
> interactions to form again. Explain how this sequence permits reshaping and recovery.

The desired response has a compact, inspectable structure:

```text
<reasoning>condition → intermediate process → outcome</reasoning>
<evidence>an exact observation quoted from the task</evidence>
<answer>a concise mechanistic conclusion</answer>
```

At inference, a user supplies a new self-contained mechanism task. No original source
document, hidden rubric, or teacher API call is needed.

## Pipeline

```text
open paper text
    │
    ├─ gpt-5.6-terra teacher: writes a self-contained how/why task
    └─ gpt-5.6-terra critic: checks every claim against the source
                    │
                    ▼
        canonical, auditable task record
            ├─ SFT view: prompt + reference completion
            └─ GRPO view: prompt + hidden causal rubric
                              │
                              ▼
                 Gemma 4 LoRA: SFT → GRPO
```

Paper-level hashing creates disjoint SFT train/validation, GRPO train/validation, and test
partitions. This prevents the same paper from appearing in both adaptation stages or the
held-out test split.

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

Authenticate with Hugging Face after accepting the Gemma model license:

```bash
hf auth login
```

Set the required OpenAI secret:

```bash
export OPENAI_API_KEY="..."
```

All non-secret settings—including models, dataset sizes, hyperparameters, paths, rewards, and
Hub repository names—are visible in each notebook's configuration cell. Do not create
additional environment variables for them.

Hugging Face uses the cached credential created by `hf auth login`. Each notebook also contains
an optional, commented `HF_TOKEN = os.environ["HF_TOKEN"]` line for environments where cached
login is unavailable. Never paste either secret into a notebook.

Then launch:

```bash
jupyter lab
```

There is no offline generation mode. Dataset authoring, criticism, and the semantic GRPO
reward use the OpenAI API and therefore incur API usage.

## Notebook order

| Notebook | Purpose | Main artifact |
|---|---|---|
| `01_generate_mechanism_sft_dataset.ipynb` | Stream open science, author tasks, run a critic, audit, and make the SFT projection | `data/processed/sft/` |
| `02_build_mechanism_grpo_dataset.ipynb` | Make GRPO splits, inspect hidden causal rubrics, and test the cached semantic judge | `data/processed/grpo/` |
| `03_finetune_sft_lora.ipynb` | Train completion-only LoRA SFT and try a new mechanism task | Local and Hub SFT adapters/checkpoints |
| `04_finetune_grpo_lora.ipynb` | Continue the SFT adapter with grouped policy optimization | Local and Hub GRPO adapters/checkpoints |

The dataset-generation notebook appends accepted and rejected records incrementally. It is
safe to interrupt and rerun.

## Dataset structure

The canonical record is the audit layer. It contains source provenance, the privately retained
source text, teacher/critic response IDs and token usage, the self-contained task, the reference
response, and an explicit causal rubric.

The training projections intentionally differ:

| Field | Canonical | SFT | GRPO |
|---|:---:|:---:|:---:|
| self-contained task/prompt | ✓ | ✓ | ✓ |
| reference completion | ✓ | ✓ | hidden |
| mechanism steps and causal links | ✓ | — | reward only |
| raw source text | ✓ | — | — |
| source license and URL | ✓ | ✓ | ✓ |

The GRPO model receives only `prompt`. TRL passes the other dataset columns directly to custom
reward functions.

## GRPO reward

The reward combines:

- exact response structure;
- evidence quoted from the self-contained task;
- required concept coverage; and
- a batched `gpt-5.6-terra` judgment of causal correctness, completeness, and evidence use.

The semantic judge is dominant because keyword overlap cannot reliably detect reversed causal
direction, missing intermediates, or plausible-sounding unsupported claims. Its structured
judgments are cached in `results/grpo_judge_cache.jsonl` using a hash of the model, prompt
version, rubric, task, and completion. Rerunning an identical judgment does not call the API
again.

Notebook 04 is intentionally single-process. A production distributed run should replace the
append-only local cache with a concurrency-safe service or database.

## Hardware behavior

`science_course.devices.detect_runtime()` selects:

1. NVIDIA CUDA, using BF16 when supported and otherwise FP16;
2. Apple MPS;
3. CPU with FP32.

The code path is portable; the workload is not equally practical on every backend. Gemma 4 E4B
LoRA training is best run on CUDA. MPS and CPU are useful for correctness checks or a smaller
model selected through `MODEL_ID`.

No CUDA-only quantization package is required. The notebooks load Gemma 4 with
`AutoModelForCausalLM`, so only its text decoder is used.

## Configuration

Each notebook begins with one configuration cell. Students can change:

- source dataset and source-count limits;
- teacher and judge model IDs;
- base policy model;
- LoRA rank, scaling, dropout, and target modules;
- SFT and GRPO hyperparameters;
- reward weights;
- checkpoint cadence and resume path;
- local output directories; and
- Hugging Face dataset/model repository IDs.

Only `OPENAI_API_KEY` is required from the environment. `HF_TOKEN` is an optional, commented
alternative to `hf auth login`.

## Hugging Face publication

Publication is enabled by default:

| Repository | Contents |
|---|---|
| [`lamm-mit/scientific-sft-grpo-data`](https://huggingface.co/datasets/lamm-mit/scientific-sft-grpo-data) | Public `sft` and `grpo` dataset configurations |
| [`lamm-mit/scientific-sft-grpo-sft`](https://huggingface.co/lamm-mit/scientific-sft-grpo-sft) | Final SFT LoRA adapter and every saved SFT checkpoint |
| [`lamm-mit/scientific-sft-grpo-grpo`](https://huggingface.co/lamm-mit/scientific-sft-grpo-grpo) | Final GRPO LoRA adapter and every saved GRPO checkpoint |

Both trainers use `hub_strategy="all_checkpoints"` and perform an explicit final push. Local
checkpoint deletion is disabled so all checkpoints remain available for upload. If a student
does not control the `lamm-mit` Hugging Face namespace, they should change the three repository
variables to their username or organization before running.

## Tested library line

The project pins the current releases checked on 2026-07-19:

- [Transformers 5.14.1](https://pypi.org/project/transformers/)
- [TRL 1.8.0](https://pypi.org/project/trl/)
- [PEFT 0.19.1](https://pypi.org/project/peft/)
- [Datasets 5.0.0](https://pypi.org/project/datasets/)
- [OpenAI Python 2.46.0](https://pypi.org/project/openai/)

Notebook 03 and notebook 04 fail early if the installed training stack is older than the tested
versions. The implementation follows current [TRL SFT](https://huggingface.co/docs/trl/en/sft_trainer)
and [GRPO](https://huggingface.co/docs/trl/en/grpo_trainer) dataset/trainer interfaces and uses
the OpenAI Responses API with
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Source dataset and licensing

The default source is
[`common-pile/peS2o`](https://huggingface.co/datasets/common-pile/peS2o), a collection of openly
licensed scientific documents with per-document license metadata. The loader preserves the
license and source URL in every generated record.

Open metadata can be imperfect. Before publishing a derived dataset, inspect source provenance,
retain per-record attribution, confirm the relevant licenses, and document the teacher/critic
procedure. The Apache-2.0 repository license covers this code; it does not replace licenses on
source documents or generated data.

## Development checks

```bash
python scripts/build_notebooks.py
python scripts/validate_notebooks.py
PYTHONPATH=src pytest
PYTHONPATH=src ruff check src tests scripts
```

Generated source data, API judgment logs, and model adapters are gitignored.

## License

The repository code and notebooks are licensed under the
[Apache License 2.0](LICENSE). Source documents and derived dataset records retain their own
per-record license and attribution metadata.

## Research caveat

These notebooks are designed for teaching a complete modern workflow. A scientific claim of
improved reasoning would additionally require expert grading, inter-rater agreement, multiple
random seeds, reward ablations, contamination checks, stronger held-out benchmarks, and
error analysis. An LLM judge is scalable supervision, not scientific ground truth.
