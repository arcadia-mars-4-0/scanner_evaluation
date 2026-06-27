# Scanner Analysis

This directory contains the reproducible analysis workflow for generating the scanner figures and tables used in the paper. The main entry point is:

```bash
.venv/bin/python analysis/generate_paper_figures.py \
  --config analysis/configs/combined_test.yaml
```

By default, the script generates both:

- `combined` outputs: diagnostic figures and tables consolidated from the scanner validation and combined-results notebooks.
- `empirical` outputs: final empirical paper figures consolidated from `scanner_paper_figures.ipynb`.

The older notebooks remain useful for exploration, but paper figure reproduction should go through `generate_paper_figures.py`.

## Common Commands

Generate the default paper outputs for the test split:

```bash
.venv/bin/python analysis/generate_paper_figures.py \
  --config analysis/configs/combined_test.yaml
```

Generate the default paper outputs for the dev split:

```bash
.venv/bin/python analysis/generate_paper_figures.py \
  --config analysis/configs/combined_dev.yaml
```

Generate only the combined diagnostic outputs:

```bash
.venv/bin/python analysis/generate_paper_figures.py \
  --config analysis/configs/combined_test.yaml \
  --figure-set combined
```

Generate only the empirical paper figures:

```bash
.venv/bin/python analysis/generate_paper_figures.py \
  --config analysis/configs/combined_test.yaml \
  --figure-set empirical
```

Write outputs somewhere other than `analysis/results/paper_figures`:

```bash
.venv/bin/python analysis/generate_paper_figures.py \
  --config analysis/configs/combined_test.yaml \
  --output-root /tmp/scanner_paper_figures
```

## Outputs

The default output root is:

```text
analysis/results/paper_figures/
```

For `analysis/configs/combined_test.yaml`, outputs are written under:

```text
analysis/results/paper_figures/combined/combined_test/
analysis/results/paper_figures/combined/combined_test_empirical/
```

The combined directory includes outputs such as:

- `grade_distribution.png`
- `grade_distribution_human.png`
- `confusion_matrices_pooled.png`
- `scanner_agreement_all_blocks.png`
- `composite_vs_validation__*.png`
- `composite_grade_distribution__*.png`
- `overview.csv`
- `pooled_metrics.csv`
- `benchmark_metrics.csv`
- `scanner_agreement_all_blocks.csv`
- `composite_metrics_all_blocks.csv`

The empirical directory includes outputs such as:

- `violation_rates_by_benchmark_max.png`
- `violation_rates_by_benchmark_floor_mean.png`
- `flag_rate_by_severity.png`
- `violation_rates_empirical_max.csv`
- `violation_rates_empirical_floor_mean.csv`
- `flag_rate_by_severity_empirical.csv`
- `empirical_sampling_strata.csv`

## Config Files

Configs live in `analysis/configs/`. A combined config has this shape:

```yaml
violation_threshold: 2
results_subdir: combined/combined_test

scanners:
  - target_scanner: ground_truth_access
    split: test
    scanner_key: null
    scan_ids: []
    validation_files:
      - post_validation_sample_MH6UxNTcbmXqXVqJnmmb7G_DS.csv
      - post_validation_sample_2WQYwt7yGUjXWvRxbhcMz8_AH.csv
```

Important fields:

- `violation_threshold`: grade threshold for a binary violation. The paper configs use `2`.
- `results_subdir`: subdirectory under the output root for combined outputs.
- `target_scanner`: scanner directory under `evals/scans/`.
- `split`: usually `dev` or `test`.
- `scanner_key`: scanner key in the parquet files. Use `null` when there is only one key.
- `scan_ids`: explicit scan IDs to include. Empty list means include all scan IDs for that scanner/split.
- `validation_files`: validation CSV files or stems to use. `null` means all validation CSVs in the validation directory.

## Benchmark Exclusions

The script excludes multiple-choice benchmarks by default:

```yaml
exclude_benchmarks:
  - gpqa_diamond
  - hle
```

This exclusion is applied to both combined and empirical outputs. To override it, add `exclude_benchmarks` to the config. To disable exclusions:

```yaml
exclude_benchmarks: []
```

## Metric Weighting

By default, metrics are IPW weighted:

```bash
--weighting weighted
```

To reproduce the unweighted combined-results behavior:

```bash
.venv/bin/python analysis/generate_paper_figures.py \
  --config analysis/configs/combined_test.yaml \
  --weighting unweighted
```

You can also set this in the config:

```yaml
metric_weighting: unweighted
```

## Empirical Figure Notes

The empirical figures use the validation-sampling design from `scanner_paper_figures.ipynb`.

Defaults:

- validation stratifier model: `gpt-5.4`
- secondary model: `sonnet-4.6`
- composite scanners: `max`, `floor_mean`
- bootstrap samples: `2000`

These can be overridden in the config:

```yaml
paper_figures:
  stratifier_model: gpt-5.4
  secondary_model: sonnet-4.6
  composites:
    - max
    - floor_mean
  boot_n: 2000
  boot_ci: 0.95
  boot_seed: 0
```

## WSL And Memory

`generate_paper_figures.py` uses a lightweight parquet loader that reads only the columns needed for plotting. This avoids loading large transcript payload columns such as `input` and `scan_events`, which can otherwise consume many GB of memory.

The script also caps common numerical thread pools to one thread by default:

```text
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

This is intentional. It keeps WSL runs predictable and avoids thread-pool behavior that can be unstable on constrained environments.

## Quick Checks

Syntax check:

```bash
.venv/bin/python -m py_compile analysis/generate_paper_figures.py
```

Dry run to a temporary output directory:

```bash
.venv/bin/python analysis/generate_paper_figures.py \
  --config analysis/configs/combined_test.yaml \
  --output-root /tmp/scanner_paper_figures_check
```

Show CLI help:

```bash
.venv/bin/python analysis/generate_paper_figures.py --help
```
