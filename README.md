# Scanner Evaluation
This repo contains the research code for the evaluation of benchmarks using scanners inspired by the [Agentic Benchmark Checklist](https://arxiv.org/abs/2507.02825).

It currently contains 4 primary sections:
- `sample_size/` contains code used in the calculation of sample size requirements performed ahead of data collection
- `analysis/` contains working analysis code, and pre-processed data supporting analysis
- `evals/` mirrors the huggingface repo, and is used to manage final data collection. 

## Evaluation Settings
Evaluation implementations and usage
Most benchmarks used an implementation from Inspect Evals, except Terminal-bench-2 which was accessed using the inspect-harbor adaptor package, and Litqa2, ScholarQA and SUPER which were from the Astabench suite. All evaluations were run with the ReAct agent. Kernelbench and Compute-eval were initially developed as non agentic single pass evaluations, but were updated and reported using agentic scaffolds.

Where possible, we chose evaluations relevant to AI Research and Development, particularly in our development set, and also evaluations actively used in model system cards. 

### Parameter Settings for evaluations
In general, maintaining consistency across evaluation task versions and inspect versions was attempted where possible, as slight differences may lead to effects on the transcripts. However variation in our transcripts for this type of study can be allowed as it may be encouraged to try and elicit and capture violation states, though ideally we want to shift these intentionally to capture any changes.

The majority of transcript files were produced with Inspect version 0.3.180.dev77, with some other versions used. CVE-bench uses two versions, with a large difference between 0.3.199 and 0.3.103 - however the task version for the eval was not changed. Swe_bench_verified had the task version change from 2-B → 3-C between transcripts, which were non-impacting changes relating to tool timeouts and configurable parameters. MLE-Bench had major task version changes - 4-B → 5-B → 5-D → 6-D, however it is only used for development. All transcripts maintained the same task version between transcripts in their test vs development set splits.
We used Inspect Harbor - v0.4.7 and a forked version of astabench.

### General requirements for producing evaluation transcripts
At their most intense, the evaluations we ran used specs equivalent to the following, which allowed us to set Inspect's --max-samples 8 (and --max-sandboxes 8 for SWE-bench), dropping MLE-bench to --max-samples 2–4
8× L40S (48 GB) 
64–128 CPU cores
512 GB–1 TB RAM (1 TB if you want MLE-bench closer to reference)
~1 TB NVMe (bump to 4 TB+ for full MLE-bench)
Linux, Docker + NVIDIA Container Toolkit, CUDA 12.x


## Environment Setup
This project uses supports [UV](https://docs.astral.sh/uv/) for environment and dependency management.

Create the project virtual environment and install dependencies:

```bash
uv venv
uv sync
```

Run project scripts through UV so they use the synced environment:

```bash
uv run python evals/hf_dataset_sync.py pull
```

## Using Scout

After installing the dependencies, Inspect Scout will be available to use in the repo.
To view the UI:

```bash
uv run scout view
```

All other relevant scout commands (including pointing at particular directories) can be found in their documentation:
https://meridianlabs-ai.github.io/inspect_scout/

## Eval Grading
The basic workflow is to use the evals directory mirror from hugging face as the source for final datasets, and the eval grading directory for work in progress grading. Scanner files should generally not be tracked in this directory while iterating. Instead, scan and validation files should be moved into evals and pushed to HF after completion. This helps maintain a separation between final data files and intermediate files. Analysis code uses the HF files as the source of truth.

Current workflow for grading:
1. Create a directory for the eval being graded
2. Create a scout.yaml file inside that directory, use the scout_template.yaml for guidance. This file should point to the `evals/` directory for transcripts, but should save scan files into the `eval_grading/<eval>/scan-results/` directory (the template follows this convention).
3. run scout scans and scout view from inside the `eval_grading/<eval>` directory. When combined with the scout.yaml this is the cleanest way to view only relevant files.
4. maintain validation csv files inside `eval_grading/<eval>/validation` until they are ready for upload to HF, then move to the appropriate folder in that directory. 
    - note: if these validation csvs are ignored by git, they will not be picked up by scout view. There is currently an exception in place to prevent this

## Syncing HF data
There is a small CLI for syncing evaluation data between the local `evals/` directory and the Hugging Face dataset `arcadia-mars-4-0/abc-scout-scanners`. The intended workflow is to use this huggingface data as the 'source of truth', while using other directories for intermediate evaluations, analysis, and scanner development.

Data is organized by evaluation (e.g., core_bench) and split into transcript logs (eval-logs), scanner results (scan-results), and validation files (validation). Data should generally be added to huggingface by copying it into these folders and pushing to remote only after it has been finalized, to prevent confusion.

The main entrypoint is:

```bash
uv run python evals/hf_dataset_sync.py
```

It uses `evals/` as the local dataset root.

### Pull Data From Hugging Face

Pull the full dataset into `evals/`:

```bash
uv run python evals/hf_dataset_sync.py pull
```

Pull a single eval subtree:

```bash
uv run python evals/hf_dataset_sync.py pull xstest
```
This downloads the remote dataset structure directly under `evals/`, for example:

```text
evals/
  xstest/
    eval-logs/
    validation/
    scan-results/
```

### Push Data To Hugging Face

Push one local eval directory back to the dataset:

```bash
uv run python evals/hf_dataset_sync.py push xstest
```

This uploads data from:

```text
evals/xstest/eval-logs/
evals/xstest/validation/
evals/xstest/scan-results/
```

If `validation/` is missing, the push still runs and logs a warning.

### Optional Scanner Namespace

By default, push uploads the full local `scan-results/` tree as-is.

If you want to place results under `scan-results/<scanner_name>/` remotely, pass `--scanner-name`:

```bash
uv run python evals/hf_dataset_sync.py push xstest --scanner-name my-scanner
```
