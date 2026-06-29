# Scanner Development

Each scanner is developed in its own subfolder, on top of a small set of shared scripts at the root of this directory.

## Layout

Shared tooling (root of `scanner_dev/`):

- [build_dataset.py](build_dataset.py) — stages a manifest-defined subset of `.eval`
  transcripts into a local `build/` directory and writes merged validation/provenance CSVs.
- [build_validation_from_scan.py](build_validation_from_scan.py) — samples a validation CSV
  (`id, target, predicate`) from a completed scan, stratified above and below a score
  threshold. The `target` column is a placeholder to be hand-labeled after export.
- [scanner_utils.py](scanner_utils.py) — shared helpers for scanners (extracting system/user
  messages, deriving task pass/fail, etc.).
- [scanner_dev_evaluation.ipynb](scanner_dev_evaluation.ipynb) — evaluate a single scanner's
  scan results against validation labels.
- [scanner_dev_comparison.ipynb](scanner_dev_comparison.ipynb) — compare two scanner runs
  (e.g. one model vs. another) on the same corpus.

Each subfolder contains:

- `<name>_scanner.py` — the scanner file.
- `config.py` — paths, manifest splits, validation columns, scan-results location, and any target rules used by the evaluation notebooks.
- `scout.yaml` — Scout config settings.
  writing scans to `./scan-results`.

## How to use

1. **Pull transcripts.** Make sure the relevant eval transcripts are present under `evals/`:

   ```bash
   uv run python tools/hf_dataset_sync.py pull
   ```

2. **Curate the corpus.** Edit the manifest for the scanner/split under
   `<scanner>/data/<split>_files.csv`. Each row points at a `.eval` file under `evals/` and
   flags whether it belongs to the dev set and/or the labeled validation subset.

3. **Stage the corpus.** Run `build_dataset.py` to copy/hard-link the selected transcripts into `build/eval-logs/` and write the merged validation and provenance CSVs:

   ```bash
   cd scanner_dev
   uv run python build_dataset.py --split test
   ```

4. **Scan.** From inside the scanner directory, run Scout against the staged corpus:

   ```bash
   cd scanner_dev/tool_failure
   uv run scout scan scout.yaml        
   uv run scout view                    
   ```

5. **Build a validation set.** Sample transcripts from a completed scan, then hand-label the `target` column:

   ```bash
   uv run python scanner_dev/build_validation_from_scan.py \
     scanner_dev/tool_failure/scan-results/scan_id=XYZ
   ```

6. **Evaluate.** Use [scanner_dev_evaluation.ipynb](scanner_dev_evaluation.ipynb) to score a
   scanner against its validation labels, or
   [scanner_dev_comparison.ipynb](scanner_dev_comparison.ipynb) to compare two runs. Both notebooks read paths from the relevant scanner's `config.py` which is set at the top of the notebook.
