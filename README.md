# Scanner Evaluation
This repo contains the research code for the evaluation of benchmarks using scanners inspired by the [Agentic Benchmark Checklist](https://arxiv.org/abs/2507.02825).

It currently contains 4 primary sections:
- `sample_size/` contains code used in the calculation of sample size requirements performed ahead of data collection
- `analysis/` contains working analysis code, and pre-processed data supporting analysis
- `eval_grading/` contains code to support human labeling of eval transcripts, and is also used as the working directory for intermediate datafiles that have not been completed and added to the huggingface repo.
- `evals/` mirrors the huggingface repo, and is used to manage final data collection. 

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
