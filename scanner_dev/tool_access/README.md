# Tool Access Dev Workflow

This directory is a staging workflow for running scanner development scans across a hand-picked set of eval transcripts drawn from multiple benchmarks.

The workflow has three parts:

1. A manifest that defines which transcripts belong in the development corpus.
2. A build step that stages those transcripts into one local directory and generates merged metadata.
3. A Scout config that scans the staged corpus.

## Files

- [data/dev_t2_files.csv](): source-of-truth manifest for the current corpus
- [build_dataset.py](): stages transcripts and generates merged metadata
- [scout.yaml](): Scout config for this staging area
- `build/`: generated outputs
- `scan-results/`: generated Scout scan outputs

## Manifest

The manifest is a CSV. Each row represents one eval log to include in the corpus.

Important columns:

- `Eval`: logical label for the run group
- `method`: run type, e.g. `default` or `Pre-training contamination`
- `eval_log_path`: repo-relative path to the `.eval` file under `evals/`
- `validation_path`: repo-relative path to the validation CSV, if one exists
- `include_in_dev`: whether the transcript should be staged by default
- `include_in_validation`: whether the transcript belongs to the labeled subset

The builder can resolve paths from `eval_file` and `graded_file`, but the current manifest already stores explicit paths and should be treated as the main control surface.

## Build

Rebuild the staging area:

```bash
uv run python scanner_dev/tool_access/build_dataset.py
```

This replaces the contents of `build/` for the currently selected subset and writes:

- `build/eval-logs/`: staged `.eval` files used as Scout transcripts input
- `dev_t5_validation.csv`: merged validation file written next to `scout.yaml`
- `build/dev_t5_provenance.csv`: provenance table linking staged logs back to source files and manifest metadata

The staged eval logs are hard-linked when possible and copied otherwise. They are real files, not symlinks, because Scout does not reliably discover symlinked eval logs in directory scans.

## Scan

From this directory:

```bash
cd scanner_dev/tool_access
uv run scout scan scout.yaml
```

If you only want a sample:

```bash
uv run scout scan scout.yaml --limit 10
```

`scout.yaml` currently points `transcripts` at `./build/eval-logs`.

## Validation

Important: Scout `validation:` does not filter which transcripts are scanned. It only evaluates scan results after the scan.

That means transcript selection happens in the build step, not in `scout scan`.

If you want validation active for the scan results, uncomment this line in [scout.yaml](/home/jeffm/projects/scanner_evaluation/scanner_dev/tool_access/scout.yaml):

```yaml
validation:
  tool_access_access: ./dev_t5_validation.csv
```

## Subsetting

You can subset the staged corpus without editing the script.

Only validated transcripts:

```bash
uv run python scanner_dev/tool_access/build_dataset.py --validated-only
```

Only specific `Eval` values:

```bash
uv run python scanner_dev/tool_access/build_dataset.py --eval mle_bench_dev --eval mle_bench_low
```

Only specific methods:

```bash
uv run python scanner_dev/tool_access/build_dataset.py --method default
```

Filters can be combined:

```bash
uv run python scanner_dev/tool_access/build_dataset.py --validated-only --method default
```

Each build replaces `build/eval-logs` with the newly selected subset.

## Multiple Corpora

If you want separate saved corpora, create additional manifests under `data/` and pass them explicitly:

```bash
uv run python scanner_dev/tool_access/build_dataset.py \
  --manifest scanner_dev/tool_access/data/contamination_round2.csv
```

This is cleaner than constantly rewriting one manifest when you want multiple recurring datasets.

## Typical Workflow

1. Edit the manifest or choose build filters.
2. Run `build_dataset.py`.
3. Check `build/dev_t5_provenance.csv` to confirm the staged subset.
4. Run `scout scan scout.yaml`.
5. Iterate on scanner logic.

## Troubleshooting

If the builder reports that a transcript or validation file is missing, pull the relevant eval subtree into `evals/` first.

Example:

```bash
uv run python evals/hf_dataset_sync.py pull mle_bench
```
