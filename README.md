# Modular Skill Internalization (MSI)

Code, configurations and synthetic training data for **Modular Skill Internalization
for Agentic Foundation Models**, covering 408 skills across TheoremQA, LogicBench,
ToolQA and MedCalc-Bench. Trained adapters and router weights are not included;
the commands below produce them. The released common data split supports
retraining, but differs from the inputs used for the paper tables.

Start with the supplied data, build anchors, train adapters, then evaluate.
Task/trajectory generation and router retraining are optional. Each stage below
lists its commands, configuration and outputs.

All examples use the existing YAML defaults. Tables list command-specific options;
[common options](#8-changing-configurations) apply across stages. A YAML key in a
table can be set in the file or passed as `--set KEY=VALUE`.

1. [Installation and models](#1-installation-and-models)
2. [Benchmark and tool resources](#2-benchmark-and-tool-resources)
3. [Synthetic data](#3-synthetic-data)
4. [Anchors](#4-anchors)
5. [Adapter training](#5-adapter-training)
6. [Retriever](#6-retriever)
7. [Inference and scoring](#7-inference-and-scoring)
8. [Changing configurations](#8-changing-configurations)

## 1. Installation and models

Open the [anonymous project page](https://anonymous.4open.science/r/MSI), download
the repository archive, and extract it. Run the Bash examples below from the
extracted directory containing `msi`, `configs/`, `SR-Agents/` and `data/packages/`. SR-Agents is bundled; no separate clone or
submodule update is needed. Python 3.10–3.12 is supported; `environment.yaml`
uses the pinned dependencies in `requirements.txt`. Use Linux, Conda and an NVIDIA driver
compatible with the installed PyTorch/vLLM CUDA builds:

```bash
conda env create -f environment.yaml
conda activate msi
python -m pip install -r requirements.txt -e ./SR-Agents -e .
cp configs/cluster.example.yaml configs/cluster.local.yaml
./msi --help
```

The supplied configurations retain the current defaults for
**NVIDIA A800-SXM4-80GB** GPUs. Full examples select four 80-GB GPUs;
adapter training also supports one training GPU plus one validation GPU. Edit `configs/cluster.local.yaml` for local paths:
`models` selects student/teacher checkpoints, `retrievers` selects embedding and
reranking models, and `paths` locates external tool resources. Relative paths are
resolved from the repository root.

Download the models used by the default pipeline, or point the cluster file to
existing local copies. Repeat this command for each row of the table, with the
matching local path; accept gated-model access terms where required:

```bash
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
```

| Public checkpoint | Default local path | Names (config / logical) | Used for |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-2B` | `models/Qwen3.5-2B` | `qwen3.5-2b` / `Qwen3.5-2B` | Student |
| `meta-llama/Llama-3.2-3B-Instruct` | `models/Llama-3.2-3B-Instruct` | `llama3.2-3b` / `Llama-3.2-3B-Instruct` | Student (gated) |
| `microsoft/Phi-3.5-mini-instruct` | `models/Phi-3.5-mini-instruct` | `phi3.5-mini` / `Phi-3.5-mini-instruct` | Student |
| `swiss-ai/Apertus-8B-Instruct-2509` | `models/Apertus-8B-Instruct-2509` | `apertus-8b` / `Apertus-8B-Instruct-2509` | Student |
| `Qwen/Qwen3.5-122B-A10B` | `models/Qwen3.5-122B-A10B` | — | Teacher; only for generating new data |
| `BAAI/bge-base-en-v1.5` | `models/bge-base-en-v1.5` | — | Anchor retrieval |
| `BAAI/bge-reranker-v2-m3` | `models/bge-reranker-v2-m3` | — | Anchor reranking |
| `BAAI/bge-m3` | `models/bge-m3` | — | Router and retrieval baseline; unnecessary with frozen routes |
| `sentence-transformers/all-mpnet-base-v2` | `models/all-mpnet-base-v2` | — | ToolQA text embedding |

The commands below use the Qwen student as the running example. For another
student, substitute its names from the table: training and evaluation select it
through `configs/paper/260-<config-name>.yaml` and
`configs/eval/<config-name>.yaml`, and `--base-model` takes the logical name.

## 2. Benchmark and tool resources

Benchmark instances are bundled in `SR-Agents/data/bench/instances/`. All MSI
stages use `data/protocol/skills.json`, the fixed 408-skill library.

| Dataset / CLI name | Test instances | Skills |
| --- | ---: | ---: |
| TheoremQA / `theoremqa` | 747 | 320 |
| LogicBench / `logicbench` | 760 | 19 |
| ToolQA / `toolqa` | 1,430 | 14 |
| MedCalc-Bench / `medcalcbench` | 1,100 | 55 |
| Total | 4,037 | 408 |

ToolQA generation, validation and inference need the official external corpus
and the text embedding model. The following commands download, unpack and place
everything in one step; `gdown` handles the Google Drive link. If Drive quota
blocks the download, fetch the archive manually from the
[link](https://drive.google.com/file/d/1zRbHzPW2x4dDcfmphBWlan8cxUCRNmqk/view)
and unpack it the same way:

```bash
hf download sentence-transformers/all-mpnet-base-v2 --local-dir models/all-mpnet-base-v2
pip install gdown
mkdir -p data/external
gdown 1zRbHzPW2x4dDcfmphBWlan8cxUCRNmqk -O external_corpus.zip
unzip -q external_corpus.zip -d data/external
mv data/external/external_corpus data/external/toolqa
rm external_corpus.zip
```

This leaves the resource directories directly inside `data/external/toolqa/`
without an extra `external_corpus/` level; keep cluster
`paths.external_dir: data/external`.

Expected resources include `flights/Combined_Flights_2022.csv`,
`coffee/coffee_price.csv`, `airbnb/Airbnb_Open_Data.csv`,
`yelp/yelp_academic_dataset_business.json`, six DBLP pickle mappings,
`agenda/agenda_descriptions_merged.jsonl` and `scirex/Preprocessed_Scirex.jsonl`.
SQL tables and embedding caches are built automatically. ToolQA generation,
training validation and inference need these resources even with supplied trajectories.

## 3. Synthetic data

### Use our generated data

The 25 archives contain **208,627 tasks and 198,717 accepted trajectories**,
covering all 408 skills with at least 480 trajectories per skill. Extract them
from the repository root to populate `data/synthetic/`:

```bash
for archive in data/packages/synthetic-*.tar.gz; do
  tar -xzf "$archive"
done
```

Set the variables used by the remaining stages; `run_id` names the output run:

```bash
run_id=paper-260
synthetic_dir=data/synthetic
demos_file=data/demos/two_shot.json
```

Continue with Section 4 using these variables in the same shell. Generating
new data is optional; expand the recipe below if needed.

<details>
<summary>Optional: generate new tasks, trajectories and demonstrations</summary>

Configure the teacher in `configs/generate.yaml` and its local path in the cluster
file. The default teacher is `Qwen/Qwen3.5-122B-A10B` (downloaded in Section 1).
Without `--api-base`, the command starts its own temporary vLLM teacher server on
the selected GPUs — tensor-parallel across them, logged to
`work/<run-id>/logs/teacher-server.log` — and shuts it down when the stage ends.
Pass `--api-base` only to reuse an existing OpenAI-compatible endpoint, in which
case GPU selection is unnecessary:

```bash
synthetic_dir=data/regenerated
run_id=regenerated-260
./msi generate --cluster configs/cluster.local.yaml --run-id "$run_id" \
  --gpu 0,1,2,3 --phase all --set paths.synthetic_dir="$synthetic_dir" --min-acceptable 480
```

| Option / YAML key | Default | What it controls |
| --- | --- | --- |
| `--phase` | `all` | `tasks`, `trajectories`, or both |
| `--teacher-model`, `--teacher-model-path` | Qwen3.5-122B-A10B | Served teacher ID and local weights |
| `--api-base` | Managed server | Existing OpenAI-compatible teacher endpoint |
| `--task-model`, `--task-api-base` | Teacher | Override task-generation model/endpoint |
| `--trajectory-model`, `--trajectory-api-base` | Teacher | Override trajectory-generation model/endpoint |
| `--evaluator-model`, `--evaluator-api-base` | Trajectory model/endpoint | Override the solution judge |
| `--teacher-tokenizer` | Teacher path/model | Local tokenizer for output budgets |
| `--min-acceptable`, `--task-oversample` | 480, 1.1 | Accepted trajectory target and candidate-task oversampling |
| `--task-temperature`, `--temperature` | 1.0, 0.7 | Task and trajectory sampling |
| `--workers`, `--delay`, `--max-rounds` | 10, 0, 16 | Concurrent requests, request delay and generation rounds |
| `--max-tokens` | 32,768 | Teacher request token budget |
| `defaults.output_max_tokens` | Per-domain table below | Visible trajectory output budget |
| `--evaluator` / `--no-evaluator` | Enabled | LLM judging; disabling uses format-only checks |
| `--dedup-threshold`, `defaults.near_threshold` | 0.88, 0.92 | Within-skill deduplication and independent benchmark-overlap filtering |
| `--retry-failed` / `--no-retry-failed` | Disabled | Retry previously recorded failures |
| `runtime.skill_lookahead`, `runtime.skill_lookahead_threshold`, `runtime.skill_lookahead_poll_seconds` | true, 32, 5 | Overlap a small remaining generation round with the next skill |
| `paths.synthetic_dir` / `--output-dir` | `data/synthetic` | Root for both task and trajectory outputs |
| `--gpu`, `--server-port`, `--gpu-memory-utilization`, `--server-max-model-len` | Selected GPUs, 8000, 0.92, 65,536 | Managed teacher server; omit GPU selection with an external endpoint |

Outputs are `$synthetic_dir/<dataset>/tasks/<skill>.json` and
`$synthetic_dir/<dataset>/trajectories/<skill>.json`. Persistent task IDs link the
two; tasks may have no accepted trajectory. Only accepted trajectories train adapters.
`--phase all` combines both stages and can top up tasks after filtering. Rerun
unchanged commands to resume; use `--retry-failed` to retry recorded failures.
Add `--skill theoremqa_000` for a pilot or `--api-base` for an existing compatible
teacher endpoint, keeping the teacher tokenizer configured locally.

For the two-shot inference baseline, render demonstrations from the new training
split using the Qwen student tokenizer:

```bash
mkdir -p "work/$run_id"
demos_file="work/$run_id/two_shot.json"
python -m msi.evaluation.demos --pool "$synthetic_dir" \
  --output "$demos_file" --tokenizer models/Qwen3.5-2B \
  --num-train 200 --num-val 80 --split-seed 42 --fixed-val-offset 400
```

| Demo option | Default / use |
| --- | --- |
| `--pool`, `--output`, `--tokenizer` | Required synthetic root, new output file and local tokenizer |
| `--num-train`, `--num-val`, `--split-seed`, `--fixed-val-offset` | 200, 80, 42, 400; match the adapter split |
| `--corpus` | `data/protocol/skills.json` |
| `--selection` | Optional existing demo file whose instance IDs should be replayed |

This selects two median-length training positives per skill, each at most 4,096
tokens. It refuses to replace existing output; skip it when already complete.

</details>

## 4. Anchors

Run both stages for the selected training split. Preparation uses BGE-base and
the reranker; answering uses the selected student and does not execute tools.

### `anchors prepare`: retrieve and assign questions

```bash
./msi anchors prepare -c configs/anchors.yaml \
  --cluster configs/cluster.local.yaml --run-id "$run_id" --gpu 0,1 \
  --set paths.synthetic_dir="$synthetic_dir" \
  --set defaults.num_train=200 --near-count 30 --random-count 30
```

| Option / YAML key | Default | What it controls |
| --- | --- | --- |
| `paths.synthetic_dir` | `data/synthetic` | Source trajectories; only the selected training positives are used |
| `defaults.num_train`, `defaults.num_val`, `defaults.split_seed`, `defaults.fixed_val_offset` | 200, 80, 42, 400 | Source split; align with adapter training |
| `--source-file` | Unset | Use an explicitly prepared synthetic source pool instead |
| `--retriever`, `--retriever-model` | `bge_base`, local BGE-base | First stage: `bge_base`, `bge_m3` or `bm25` |
| `--rerank` / `--no-rerank`, `--reranker-model` | Enabled, local BGE reranker | Apply or disable second-stage reranking |
| `--first-stage-top-k`, `--top-k` | 10, 10 | Candidate and final ranking sizes |
| `--near-count`, `--random-count`, `--oversample` | 30, 30, 1.6 | Desired anchor counts and extra answer candidates |
| `--retrieval-batch-size`, `--retrieval-query-chunk-size`, `--retrieval-dtype` | 256, 4096, `bfloat16` | Retrieval throughput and dtype (`float32`/`float16`/`bfloat16`) |
| `--reranker-batch-size`, `--reranker-max-length` | 32, 512 | Reranker batch and sequence length |
| `--gpu` | Selected GPUs | Retrieval devices |
| `paths.anchor_task_pool`, `paths.anchor_assignment_dir`, `paths.anchor_retrieval_dir` | Run-scoped paths in YAML | Prepared questions, assignments and rankings |

Outputs are under `data/anchor_tasks/<run-id>/` and
`data/anchor_retrieval/<run-id>/`. The default source builder excludes all skills'
validation questions. Prepare once for each split/quota.

### `anchors answer`: generate student answers

```bash
./msi anchors answer -c configs/anchors.yaml \
  --cluster configs/cluster.local.yaml --run-id "$run_id" --gpu 0 \
  --base-model Qwen3.5-2B --set paths.synthetic_dir="$synthetic_dir"
```

| Option / YAML key | Default | What it controls |
| --- | --- | --- |
| `--base-model` | Required | Logical student name from `models`; repeat answering for each student |
| `--served-model`, `--api-base` | Selected model, managed server | Override served ID or use an existing endpoint |
| `--base-answer-max-tokens`, `--workers` | 16,384, 128 | Answer token limit and concurrency |
| `paths.anchor_task_pool`, `paths.anchor_assignment_dir` | Same run as preparation | Questions, assignments and their stored quotas |
| `paths.synthetic_dir` | `data/synthetic` | Trajectories for the post-answer input audit |
| `paths.anchor_dir` / `--output-dir` | `data/anchors/<run-id>` | Root for model-specific anchor outputs |
| `--gpu`, `--server-port`, `--gpu-memory-utilization`, `--server-max-model-len` | Selected GPU, 8000, 0.92, 32,768 | Managed student server |

Output: `data/anchors/<run-id>/<base_model>/<dataset>/<skill>.json`.
Answering reads quotas from assignments; set them during `prepare`. Insufficient
candidates/answers stop construction. Anchors are used only in training.

## 5. Adapter training

Use the model, budget and text-condition configuration you want to train:

```bash
for regime in notext withtext; do
  ./msi train -c configs/paper/260-qwen3.5-2b.yaml \
    --cluster configs/cluster.local.yaml --run-id "$run_id" --gpu 0,1,2,3 \
    --set paths.synthetic_dir="$synthetic_dir" --regime "$regime" \
    --anchors --num-train 200 --num-val 80 --val-offset 400 --split-seed 42 \
    --rank 8 --learning-rate 1e-4 --epochs 8
done
```

`notext` removes skill text from training prompts; `withtext` keeps it. The command
covers all 408 skills. Add `--skill theoremqa_000 --dry-run` to inspect one job;
remove `--dry-run` for an actual single-skill run. Domain subsets use repeated
`--dataset` flags. Two GPUs (`--gpu 0,1`) support one worker and one validator;
four support three independent workers and one validator.

| Option / YAML key | Default in the example | What it controls |
| --- | --- | --- |
| `--regime` / `regimes` | Selected paper file's regime | `notext` removes skill text; `withtext` keeps it. Repeat the flag to select both. This controls **training**, not inference. |
| `--base-model` / `model.path` | Student selected by config/cluster | Local base weights; use the matching model config for architecture/LoRA targets |
| `paths.synthetic_dir`, `paths.anchor_dir` | Selected pool, `data/anchors/<run-id>` | Positive and student-specific anchor inputs |
| `--anchors` / `--no-anchors` | Enabled | Include or omit anchors |
| `--num-train`, `--num-val`, `--split-seed` | 200, 80, 42 | Positive/validation counts and shuffle seed |
| `--val-offset`, `defaults.val_fix_val` | 400, true | Fixed validation slice `[400:480]`; disabling fixed validation uses the slice after the training prefix |
| `defaults.anchor_near_count`, `defaults.anchor_random_count` | 30, 30 | Anchor subset counts; must be available in generated anchors |
| `--rank`, `--learning-rate`, `--epochs` | 8, `1e-4`, 8 | LoRA rank and optimization |
| `--batch-size`, `--gradient-accumulation-steps` | 1 × 8 for Qwen 260 | Effective batch; other model/skill layouts are in YAML |
| `--max-length` | 20,480 | Maximum training sample length |
| `--val-backend` | `vllm` | `vllm` uses a separate validation GPU; `inprocess` validates synchronously on the training GPU. Both use 1-SE epoch selection. `none` disables validation. |
| `--val-freq`, `--val-max-tokens` | 2, per-domain table | Validation epochs and generation budget |
| `--val-workers`, `--val-concurrency` | 512, 4 | Validation requests and concurrent validation jobs |
| `--val-async` / `--no-val-async` | Async | Validation scheduling |
| `--compute-val-loss` / `--no-compute-val-loss` | Disabled | Additional validation-loss computation |
| `--base-baseline` / `--no-base-baseline` | Disabled | Base-model synthetic-validation diagnostic |
| `--no-validation` | Not set | Exploratory training without generation validation/1-SE selection; still uses `--num-train` positives |
| `--gpu`, `--val-gpu` | Four selected devices; last used for validation | Training devices and an optional explicit validation GPU |
| `--api-base`, `--val-port` | Managed validator, 8003 | Existing validation endpoint or managed-server port |
| `--gpu-memory-utilization`, `--server-max-model-len` | 0.92, 65,536 | Validation serving limits |
| `paths.adapter_dir` / `--output-dir` | `work/<run-id>/adapters` | Adapter output root |

`configs/paper/260-<model>.yaml` provides one training configuration per model —
the paper's nominal 260-record budget (200 positives + 60 anchors), with both text regimes selectable through
`--regime`. Other data budgets repeat the same commands with the quota table
below and a distinct run ID; anchors must be prepared with the same counts
(Section 4). `configs/train/<model>.yaml` exposes the same interface with
generic defaults. ToolQA trajectories can expand into multiple training
samples; record quotas are not token/step counts.

| Budget | Positives (`num_train`) | Near-miss anchors | Random anchors |
| ---: | ---: | ---: | ---: |
| 64 | 48 | 8 | 8 |
| 160 | 122 | 19 | 19 |
| 260 | 200 | 30 | 30 |
| 384 | 300 | 48 | 36 |
| 512 | 400 | 64 | 48 |

For example, the 512 budget for Qwen:

```bash
./msi anchors prepare -c configs/anchors.yaml \
  --cluster configs/cluster.local.yaml --run-id paper-512 --gpu 0,1 \
  --set paths.synthetic_dir=data/synthetic \
  --set defaults.num_train=400 --near-count 64 --random-count 48
./msi anchors answer -c configs/anchors.yaml \
  --cluster configs/cluster.local.yaml --run-id paper-512 --gpu 0 \
  --base-model Qwen3.5-2B --set paths.synthetic_dir=data/synthetic
./msi train -c configs/paper/260-qwen3.5-2b.yaml \
  --cluster configs/cluster.local.yaml --run-id paper-512 --gpu 0,1,2,3 \
  --set paths.synthetic_dir=data/synthetic --regime notext \
  --set defaults.num_train=400 \
  --set defaults.anchor_near_count=64 --set defaults.anchor_random_count=48
```

Validation stays at 80 records from offset 400 in seed-42 order for every
budget, and the positive prefixes are nested (the 64-budget positives are a
prefix of the 512-budget positives).

Validation generates answers at epochs 2/4/6/8. Selection chooses the earliest
evaluated epoch within one binomial standard error of the best validation accuracy,
not necessarily epoch 8. Adapters are saved to
`work/<run-id>/adapters/<base_model>/<notext|withtext>/<skill>/`.
Rerunning an unchanged command resumes incomplete jobs. Full-domain LoRA inference
requires the complete corresponding adapter pool, not just a single trained skill.

## 6. Retriever

Skip this section when using the supplied frozen routes in `data/routing/`.
To train a new BGE-M3 router, run `prepare`, `train`, then `retrieve` below.

### `retriever prepare`: construct train/dev data

First finish anchor preparation in Section 4. The retriever uses its source
questions, skill labels and retrieval rankings to construct hard negatives:

```bash
router_run=retriever-new
./msi retriever prepare -c configs/retriever.yaml \
  --cluster configs/cluster.local.yaml --run-id "$router_run" \
  --set paths.task_pool="data/anchor_tasks/$run_id/task-pool.json" \
  --set paths.anchor_retrieval="data/anchor_retrieval/$run_id/bge_base_rerank.json"
```

| Configuration key | Default | What it controls |
| --- | --- | --- |
| `paths.task_pool`, `paths.anchor_retrieval` | `paper-260` anchor outputs | Synthetic questions and hard-negative rankings |
| `paths.corpus`, `paths.instances_dir` | Bundled skills/benchmarks | Candidate skills and independent overlap exclusion |
| `defaults.seed`, `defaults.dev_ratio` | 42, 0.1 | Reproducible train/dev split |
| `defaults.near_threshold` | 0.92 | Benchmark and train/dev near-overlap exclusion |
| `--work-dir`, `--run-id` | `work`, chosen run | Output is `work/<router_run>/data.json`; use a new run to prepare different data |

Preparation runs on CPU; `--gpu` is not required. It prepares the complete corpus,
not a `--dataset`/`--skill` subset.

### `retriever train`: learn the router

```bash
./msi retriever train -c configs/retriever.yaml \
  --cluster configs/cluster.local.yaml --run-id "$router_run" --gpu 0,1,2,3 \
  --set paths.prepared_data="work/$router_run/data.json"
```

| Configuration key / option | Default | What it controls |
| --- | --- | --- |
| `paths.prepared_data` | `data/retriever/data.json` | Training input; explicitly override for newly prepared data as above |
| `model.path` | Local BGE-M3 | Initial checkpoint, also configurable through cluster `retrievers.bge_m3` |
| `defaults.epochs`, `defaults.batch_size` | 3, 16 | Epochs and global batch (16 in the paper protocol); the batch must divide evenly across the selected GPUs |
| `defaults.learning_rate`, `defaults.warmup_ratio` | `2e-5`, 0.1 | Optimization schedule |
| `defaults.temperature`, `defaults.max_length` | 0.05, 8192 | Contrastive temperature and sequence length |
| `defaults.seed`, `defaults.eval_batch_size` | 42, 16 | Training seed and dev-retrieval batch |
| `defaults.smoke_steps`, `defaults.dev_limit` | 0, 0 | Optional limited-run controls; zero uses the full run |
| `--gpu` | Selected devices | Supports 1, 2 or 4 GPUs |

Selection uses domain-macro dev Recall@1; ties retain the earlier epoch. Checkpoint
selection is recorded in `work/<router_run>/selected.json`. Training uses all rows
in the prepared input; `--dataset`/`--skill` do not subset it.

Trained router weights are not distributed, and neither are trained adapters —
both are produced by the commands above. The paper's actual inference-time
routing is already frozen in `data/routing/<dataset>.json` (consumed by the
default `paper_ft` profile), so evaluation does not need router weights. We do
include the historical router training input (187,763 train / 20,858 dev
queries). Extract it from the repository root to create
`data/retriever/data.json`:

```bash
tar -xzf data/packages/retriever.tar.gz
```

To use it instead of `prepare`, set `router_run=retriever` and
`paths.prepared_data=data/retriever/data.json` in the training command above.

### `retriever retrieve`: rank benchmark skills

```bash
./msi retriever retrieve -c configs/retriever.yaml \
  --cluster configs/cluster.local.yaml --run-id "$router_run" --gpu 0
```

| Configuration key / option | Default | What it controls |
| --- | --- | --- |
| `paths.selected_checkpoint` | `work/<router_run>/selected.json` | Trained checkpoint selection record |
| `paths.corpus`, `paths.instances_dir` | Bundled skills/benchmarks | Candidate library and query datasets |
| `--dataset` | All four | Repeat to select domains |
| `defaults.eval_batch_size`, `--gpu` | 16, selected device | Retrieval batch and device |
| `paths.output_dir` / `--output-dir` | `results/retrieval/<router_run>` | Rankings and Recall@1/@5/@10/nDCG per dataset |

### Configure other retrieval methods

Eval YAMLs expose BM25, BGE-base, BGE-M3 and reranking under `retrieval_profiles`.
Set `retriever`, `model`, `top_k` and `batch_size` for a first-stage profile;
`source_profile`, `reranker_model` and `max_length` configure reranking. The
`method_retrieval` mapping chooses which profile an inference method consumes.

| Retrieval profile key | Meaning |
| --- | --- |
| `retriever`, `model`, `top_k`, `batch_size` | First-stage implementation, local model, ranking size and batch |
| `source_profile`, `reranker_model`, `max_length` | Rerank an existing first-stage profile |
| `frozen`, `required_profile` | Read precomputed routes and check their declared retrieval identity |
| `checkpoint_record` | Selected checkpoint for the `bge_m3_ft_synthetic` profile |
| `paths.retrieval.<profile>.<dataset>` | Ranking file for each profile/domain |
| `method_retrieval.<method>` | Connect an inference method to a retrieval profile |

For retrieval metrics alone, without language-model inference:

```bash
sragents retrieve --retriever bm25 --corpus data/protocol/skills.json \
  --instances SR-Agents/data/bench/instances/theoremqa.json \
  --top-k 10 --output results/retrieval/theoremqa/bm25.json
```

| `sragents retrieve` option | What it controls |
| --- | --- |
| `--retriever` | Retrieval implementation, e.g. `bm25`, `bge`, `bge_m3` |
| `--corpus`, `--instances`, `--output` | Skill library, query file, output ranking file |
| `--top-k` | Number of ranked skills per query (10 in the example) |
| `--retriever-arg KEY=VALUE` | Repeat for model-specific settings such as `model_path` and `batch_size` |

Use `bge` or `bge_m3` with `--retriever-arg model_path=models/bge-base-en-v1.5`
or `models/bge-m3` for dense retrieval. Choose the dataset's instance file and a
distinct output filename. All methods use the same 408-skill corpus.

## 7. Inference and scoring

After training the required adapters, evaluate with the supplied model config:

```bash
./msi evaluate -c configs/eval/qwen3.5-2b.yaml \
  --cluster configs/cluster.local.yaml --run-id "$run_id" --gpu 0,1,2,3 \
  --set demos_file="$demos_file"
```

This runs the default eight methods on all four domains and scores each output
automatically. Use `--dataset theoremqa` for one domain and repeat `--method` to
select methods. Direct and skill-text methods need no adapters, so an initial
setup check can run before training:

```bash
./msi evaluate -c configs/eval/qwen3.5-2b.yaml \
  --cluster configs/cluster.local.yaml --run-id baseline-check --gpu 0 \
  --dataset theoremqa --method naive --method golden_skill --dry-run
```

Remove `--dry-run` to infer and score all 747 TheoremQA instances. The pipeline
starts its model servers automatically.

| Method | Adapter training | Inference text / routing |
| --- | --- | --- |
| `naive` | None | No skill |
| `golden_skill`, `golden_skill_2shot` | None | Gold skill; optionally two synthetic demos |
| `golden_lora` | `notext` | No skill text; gold adapter |
| `golden_lora_text` | `withtext` | Gold skill text and adapter |
| `golden_lora_notext_text` | `notext` | Gold adapter; skill text added at inference only |
| `bm25_top1` | None | Retrieved skill text via BM25 (`bm25` profile) |
| `bge_base_top1` | None | Retrieved skill text via BGE-base (`bge_base` profile) |
| `bge_base_rerank_top1` | None | Retrieved skill text via BGE-base + reranker (`bge_base_rerank`) |
| `bge_m3_top1` | None | Retrieved skill text; mapped to the fine-tuned router's frozen routes (`paper_ft`) by default, raw BGE-M3 (`bge_m3`) if remapped |
| `bge_m3_rerank_top1` | None | Retrieved skill text via BGE-M3 + reranker (`bge_m3_rerank`) |
| `retrieved_lora` | `notext` | No skill text; retrieved adapter (`paper_ft`) |
| `retrieved_lora_text` | `withtext` | Retrieved skill text and adapter (`paper_ft`) |
| `retrieved_lora_notext_text` | `notext` | Retrieved adapter; skill text added at inference only |
| `mismatched_lora` | `notext` | Same-domain non-gold adapter; frozen seed-101 route by default; gold tools |

`method_retrieval` determines the actual router: the default `bge_m3_top1` and
retrieved-LoRA methods share frozen fine-tuned `paper_ft` routes. For other
retrievers, select e.g. `--method bm25_top1 --method bge_base_top1`, or set
`method_retrieval.bge_m3_top1=bge_m3` to use raw BGE-M3.

<details>
<summary>Point evaluation to your newly trained router</summary>

Use `$router_run` from Section 6 and evaluate into a new run while reusing the
trained adapters. The same profile/path pattern can be set in the eval YAML.

```bash
for dataset in theoremqa logicbench toolqa medcalcbench; do
  ./msi evaluate -c configs/eval/qwen3.5-2b.yaml \
    --cluster configs/cluster.local.yaml --run-id "${run_id}-${router_run}" --gpu 0 \
    --adapter-dir "work/$run_id/adapters" --dataset "$dataset" \
    --method bge_m3_top1 --method retrieved_lora --method retrieved_lora_text \
    --set method_retrieval.bge_m3_top1=bge_m3_ft_synthetic \
    --set method_retrieval.retrieved_lora=bge_m3_ft_synthetic \
    --set method_retrieval.retrieved_lora_text=bge_m3_ft_synthetic \
    --set "retrieval_profiles.bge_m3_ft_synthetic={checkpoint_record: work/$router_run/selected.json, batch_size: 16}" \
    --set "paths.retrieval.bge_m3_ft_synthetic.$dataset=results/retrieval/$router_run/$dataset.json"
done
```

</details>

| Option / YAML key | Default | What it controls |
| --- | --- | --- |
| `--method` / `methods` | Eight methods in YAML | Repeat to select inference conditions from the table above |
| `method_adapters.<method>` | `notext` or `withtext` in YAML | Which trained adapter regime to load; inference text is determined by the **method** |
| `--base-model` / `model.path` | Student selected by config/cluster | Local base checkpoint |
| `--adapter-dir` | `work/<run-id>/adapters` | Reuse adapters from another training run |
| `method_retrieval`, `retrieval_profiles`, `paths.retrieval` | `paper_ft` for default retrieved methods | Profile selection, retriever settings and per-domain route files |
| `--retrieval-file` | Unset | Explicit route file; use with the corresponding `--dataset` and matching profile metadata |
| `demos_file` | `data/demos/two_shot.json` | Demo payload; set to your generated file when using new synthetic data |
| `--temperature`, `--max-tokens`, `--workers` | 0, per-domain table, 128 | Decoding and request concurrency |
| `--lora-backend` | `vllm` | `vllm` or `inprocess` adapter loading |
| `--fallback` | `error` | Missing adapters/routes stop inference; `allow` explicitly permits fallback |
| `--max-loaded-loras` | 16 | Maximum loaded adapters per serving wave |
| `--api-base` | Managed server | Existing compatible inference endpoint |
| `--gpu`, `--server-port`, `--gpu-memory-utilization`, `--server-max-model-len` | Selected devices, 8100, 0.92, 65,536 | Managed inference servers |
| `--eval-workers` | 32 | Scoring concurrency |
| `--output-dir`, `paths.eval_dir` | `results/inference/<run-id>`, `results/eval/<run-id>` | Prediction and score output roots |

Predictions are saved to
`results/inference/<run-id>/<dataset>/<base_model>/<method>.jsonl`;
scores to `results/eval/<run-id>/<dataset>/<base_model>/<method>.json`.
Score files contain `metrics.correct`, `metrics.total`, `metrics.accuracy` and
per-instance `details`.

Overall accuracy is **sum(correct) / sum(total)** across all 4,037 instances,
not the mean of four percentages. Report domain and per-skill scores as well.
Oracle routing uses gold skills and must be distinguished from retrieved routing.

## 8. Changing configurations

Keep machine paths in `configs/cluster.local.yaml` and experiment settings in the
stage YAMLs. Common command options are listed once here; stage-specific tables
above give the remaining controls.

| Common option | Meaning and scope |
| --- | --- |
| `-c` / `--config` | Stage YAML; defaults to the corresponding file in `configs/` |
| `--cluster` | Local model/resource paths and interpreter settings |
| `--run-id` | Run name used in `{run_id}` paths; resolved configs go under `work/<run-id>/resolved/` |
| `--dataset` | Repeat for domains in generation, anchors, adapter training, inference or retriever retrieval |
| `--skill` / `--skills` | Exact skills for generation, anchors and adapter training; comma-separated or repeated. Not an evaluation/retriever subset control. |
| `--work-dir`, `--cache-dir` | Override working and cache roots |
| `--output-dir` | Stage output root as listed above; not available for `anchors prepare`, and not the prepared/trained retriever work directory |
| `--set KEY=VALUE` | Repeat for nested YAML fields; values are parsed as YAML |
| `--model-param MODEL.KEY=VALUE` | Scoped defaults for a logical model |
| `--dataset-param DATASET.KEY=VALUE` | Scoped defaults for a dataset |
| `--skill-param SKILL.KEY=VALUE` | Scoped defaults for a skill |
| `--dry-run`, `--help` | Inspect commands without running a stage, or list its CLI |

For generation/anchors/adapter training/evaluation, settings resolve from protocol
defaults → YAML defaults → model → dataset → skill overrides; direct flags such
as `--batch-size` take precedence. Retriever settings use its `defaults` directly,
so set them with `--set defaults.KEY=VALUE`. A complete override example:

```bash
./msi train -c configs/paper/260-qwen3.5-2b.yaml \
  --cluster configs/cluster.local.yaml --run-id "${run_id}-custom" --gpu 0,1 \
  --set paths.synthetic_dir="$synthetic_dir" --set paths.anchor_dir="data/anchors/$run_id" \
  --regime notext --batch-size 2 --gradient-accumulation-steps 4 \
  --skill theoremqa_000 --dry-run
```

Use a new run ID when changing data, weights, prompts or routes. Input quotas
and the validation split must agree between anchors and adapter training.
`--adapter-dir` reuses a trained pool for a new evaluation run. Inference uses one
server per GPU; teacher generation uses tensor parallelism; retriever training
supports 1/2/4 GPUs. Serving memory utilization defaults to 0.92. Adjust hardware
settings as needed while keeping experimental quantities explicit.

| Domain | Teacher visible-output limit | Validation/inference limit |
| --- | ---: | ---: |
| TheoremQA | 10,240 | 10,240 |
| LogicBench | 8,192 | 8,192 |
| ToolQA | 6,144 cumulative | 16,384 |
| MedCalc-Bench | 8,192 | 8,192 |

The teacher request budget is 32,768 tokens. Shared prompt/token rules and thinking
modes are in `src/msi/protocol.py`. Validation selects checkpoints on synthetic
positives; benchmark labels must not select epochs or routing settings.

## Paper experiment recipes

All rows use the installation and resource setup above. New-data runs are not
exact reruns of historical scores; see the reproduction notes below.

| Paper experiment | Configuration and command selection |
| --- | --- |
| Oracle comparison | Section 5, both regimes, then Section 7 with `naive`, `golden_skill`, `golden_skill_2shot`, `golden_lora`, `golden_lora_text`; repeat for each of the four model configs |
| Retrieved comparison | Same adapters; Section 7 with `bge_m3_top1`, `retrieved_lora`, `retrieved_lora_text`; defaults share frozen `paper_ft` routes |
| Skill specificity | `mismatched_lora`, same-domain non-gold routes at seeds 101/202/303; example below |
| Data-budget curve | Qwen and Llama, both regimes, each budget in Section 5; prepare **and answer** matching anchors in a distinct run |
| Retriever comparison | Section 6 retrieval profiles and Recall@K outputs; Section 7 uses the same profile for text and adapter methods when comparing downstream accuracy |

For the skill-specificity comparison, reuse the trained `notext` adapter library
and evaluate each frozen random seed separately:

```bash
for seed in 101 202 303; do
  for dataset in theoremqa logicbench toolqa medcalcbench; do
    ./msi evaluate -c configs/eval/qwen3.5-2b.yaml \
      --cluster configs/cluster.local.yaml --run-id "${run_id}-mismatch-s${seed}" --gpu 0 \
      --adapter-dir "work/$run_id/adapters" --dataset "$dataset" \
      --method mismatched_lora \
      --retrieval-file "data/routing/random/${dataset}_s${seed}.json"
  done
done
```

`mismatched_lora` defaults to seed 101 when no route file is supplied. Keep the
gold tool condition and compare against `golden_lora`; report the three-seed
mean and variability. Random routing seeds are not independent training seeds.

## Reproduction notes

<details>
<summary>Validation status, data provenance and limits of exact reproduction</summary>

CPU/interface checks, all 408 skills' five-budget splits, archive contents and
command dry-runs have been checked. A fresh GPU end-to-end run, complete paper
tables and the final anonymous download have not been validated. Some historical
model revisions and external resource versions are not pinned; downloading current
upstream weights does not establish exact checkpoint equivalence.

The release keeps the complete synthetic pool, including tasks without an
accepted trajectory and trajectories beyond the requested quota. Schema prefixes
and model metadata are normalized; 20 author-specific interpreter prefixes in
ToolQA tracebacks are replaced with `/usr/local`. Other messages, questions, IDs
and order are preserved. Two-shot demos use the two median-length training
positives per skill under the Qwen tokenizer, with a 4,096-token per-demo cap.

Question-field exact/near overlap checks are part of preparation and training;
they do not establish absence of semantic overlap. A prior historical-input
audit found an overlapping determinant problem. The common pool, fixed validation
slice and regenerated anchors differ from historical runs, so new scores may differ.

The ToolQA Agenda corpus used historically has 9,495 rows; another available copy
has 9,494 matching initial rows. The extra row describes a Jessica meeting on
2022/11/28, 09:00 AM–06:00 PM, with no ID. Exact official-package provenance of
that row remains unresolved. SQL currently has a cancellable 30-second budget;
historical runs did not all share that budget. Keep resource versions, prompts,
token limits, text conditions and routes fixed when making matched comparisons.

</details>

## Licenses and sources

<details>
<summary>Code licenses, data attribution and model terms</summary>

The root [MIT license](LICENSE) applies to MSI code. Dependencies, data and
models retain their own terms; upstream attribution does not identify the
authors of this submission.

- SR-Agents: MIT; copyright and full text in `SR-Agents/LICENSE`, with the
  copyright line anonymized for review. The bundled version uses four
  domains, the fixed 408-skill corpus, explicit model paths and patched
  tool execution.
- [ToolQA](https://github.com/night-chen/ToolQA): Apache-2.0, retained in
  `SR-Agents/LICENSE-ToolQA`. Its ReAct engine, prompts, examples and tools are
  adapted for chat providers, token/step limits, concurrency, local resources,
  cancellable in-memory SQL and shared embedding caches.
- [TheoremQA](https://github.com/TIGER-AI-Lab/TheoremQA): reference evaluation
  rules adapted to the bundled evaluator; MIT, copyright 2024 TIGER Lab, full
  text in `SR-Agents/LICENSE-TheoremQA`.
- [MedCalc-Bench](https://github.com/ncbi-nlp/MedCalc-Bench): source of the
  attributed extraction/scoring rules. A separate upstream code license has
  not been established; the data license alone is not an independent grant for
  evaluator code. This permission boundary remains unresolved.
- The bundled skill library carries an aggregate MIT label that does not
  override TheoremQA,
  [LogicBench](https://github.com/Mihir3009/LogicBench), ToolQA or
  [MedCalc-Bench](https://huggingface.co/datasets/ncbi/MedCalc-Bench) component terms.
  MedCalc-Bench data are attributed to its creators and NLM/DIR BioNLP Group,
  under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en).
  Applicable derivatives retain attribution and the same license; generated
  records add questions/answers/messages and the provenance edits stated above.
- ToolQA corpus files are not bundled. Recorded tool observations in synthetic
  trajectories and demonstrations remain subject to their source terms,
  including Yelp, DBLP and SciREX. Their availability is not a blanket
  redistribution permission, and MSI does not relicense those observations.
- Model weights are not bundled; use their upstream license and access terms.

</details>
