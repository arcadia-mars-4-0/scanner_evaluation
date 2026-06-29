# Automated Transcript Analysis for Detecting Flaws in Agentic Benchmarks

Paper Link: TBC

Research code for detecting evaluation flaws s using *scanners*
inspired by the [Agentic Benchmark Checklist](https://arxiv.org/abs/2507.02825). A scanner
inspects agent transcripts for a specific failure mode (e.g. accessing ground-truth answers,
guessing, tool-failure handling, answer formatting), and we measure each scanner against post-validated labels to estimate how reliably benchmark violations can be detected.

## Data Provenance

The dataset can be found at is the Hugging Face dataset
`arcadia-mars-4-0/abc-scout-scanners` 

Pull it with:
```bash
uv run python tools/hf_dataset_sync.py pull
```

The analysis pipeline reads scan results and validation files from
`evals/scans/<scanner>/<split>/` (e.g. `evals/scans/ground_truth_access/test/`), which a full
pull populates.

## Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

Create the project virtual environment and install dependencies:

```bash
uv venv
uv sync
```

Run project scripts through uv so they use the synced environment:

```bash
uv run python tools/hf_dataset_sync.py pull
```

Exact dependency versions are pinned in [pyproject.toml](pyproject.toml) and locked in
`uv.lock` for reproducibility.

## Evaluation Settings

Most benchmarks used an implementation from Inspect Evals, except Terminal-bench-2 (accessed
via the inspect-harbor adaptor package) and Litqa2, ScholarQA, and SUPER (from the Astabench
suite). All evaluations were run with the ReAct agent. Kernelbench and Compute-eval were
initially developed as non-agentic single-pass evaluations, then updated and reported using
agentic scaffolds.

Where possible, we chose evaluations relevant to AI research and development — particularly in
our development set — and evaluations actively used in model system cards.

### Parameter settings

We aimed for consistency across evaluation task versions and Inspect versions where possible,
since differences can affect transcripts. Some variation is acceptable for this kind of study,
as it can help elicit and capture violation states.

The majority of transcript files were produced with Inspect `0.3.180.dev77`, with some other
versions used. CVE-bench uses two versions, with a large difference between `0.3.199` and
`0.3.103`, though the task version was not changed. SWE-bench Verified had its task version
change from 2-B → 3-C between transcripts (non-impacting changes to tool timeouts and
configurable parameters). MLE-Bench had major task version changes (4-B → 5-B → 5-D → 6-D), but
it is only used for development. All transcripts kept the same task version within their test
vs. development splits. We used Inspect Harbor v0.4.7 and a forked version of Astabench.

### Hardware

At their most intense, the evaluations used specs equivalent to the following, which allowed
Inspect's `--max-samples 8` (and `--max-sandboxes 8` for SWE-bench), dropping MLE-bench to
`--max-samples 2–4`:

- 8× L40S (48 GB)
- 64–128 CPU cores
- 512 GB–1 TB RAM (1 TB to get MLE-bench closer to reference)
- ~1 TB NVMe (4 TB+ for full MLE-bench)
- Linux, Docker + NVIDIA Container Toolkit, CUDA 12.x

## Using Scout

After installing dependencies, Inspect Scout is available in the repo. To view the UI:

```bash
uv run scout view
```

### Pull

Pull the full dataset into `evals/`:

```bash
uv run python tools/hf_dataset_sync.py pull
```

Pull a single eval subtree:

```bash
uv run python tools/hf_dataset_sync.py pull core_bench
```

This downloads the remote dataset structure directly under `evals/`, for example:

```text
evals/
  core_bench/
    eval-logs/
    validation/
    scan-results/
```
