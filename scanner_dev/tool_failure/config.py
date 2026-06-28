from pathlib import Path

_DIR = Path(__file__).parent          # scanner_dev/tool_failure/
_SCANS_DIR = Path(__file__).parents[2] / "evals/scans/tool_failure/dev/scan-results"

# build dataset config
REPO_ROOT = _DIR.parents[1]
DEFAULT_BUILD_DIR = _DIR / "build"
SPLITS = {
    "dev": {
        "manifest": _DIR / "data" / "dev_t2_files.csv",
        "validation": _DIR / "dev_t2_validation.csv",
        "provenance": "dev_t2_provenance.csv",
    },
    "test": {
        "manifest": _DIR / "data" / "test_t2_files.csv",
        "validation": _DIR / "test_t2_validation.csv",
        "provenance": "test_t2_provenance.csv", # Note: these files don't exist yet since we haven't built the test dataset, but we'll want to follow the same naming convention when we do.
    },
}
VALIDATION_COLUMNS = ("id", "target", "predicate")

# ── Shared dataset inputs ──────────────────────────────────────────────────────

PROVENANCE_CSV = _DIR / "build/dev_t2_provenance.csv"
VALIDATION_CSV = _DIR / "dev_t2_validation.csv"

VIOLATION_THRESHOLD = 1

INCLUDE_EVAL_FILES: list[str] | None = None
EXCLUDE_EVAL_FILES: list[str] | None = None

# ── Single-scanner evaluation (scanner_dev_evaluation.ipynb) ──────────────────

SCAN_RESULTS_PATH = _SCANS_DIR / "scan_id=JfL9Q2bcaXtpiZZvcQRV7a"
SCANNER_KEY = "tool_failure"

# ── Two-scanner comparison (scanner_dev_comparison.ipynb) ─────────────────────

SCANNERS: list[dict] = [
    {
        "label": "GPT 5.4",
        "scan_results_path": _SCANS_DIR / "scan_id=JfL9Q2bcaXtpiZZvcQRV7a",
        "scanner_key": "tool_failure",
    },
    {
        "label": "Claude Sonnet 4.6",
        "scan_results_path": _SCANS_DIR / "scan_id=cpvwU5davvTxGbje6jdqYt",
        "scanner_key": "tool_failure",
    },
]

# ── Target rules ───────────────────────────────────────────────────────────────

TARGET_RULES: dict[str, dict] = {
    # Human-labeled validation runs — use the merged validation CSV.
    "2026-03-05T07-19-51+00-00_terminal-bench-2-0_EFZcEhAZEwRExw66VnVjVD.eval": {"mode": "validation"},
    "2026-03-07T15-51-32+00-00_mle-bench_B4Erv6ATfMjJz6dQFXh3AH.eval": {"mode": "validation"},
    "2026-03-21T09-34-42+00-00_mle-bench_awRHnzf67AWSZYtXs7q2bf.eval": {"mode": "validation"},
    "2026-03-31T10-32-04+00-00_mle-bench_XNQHMCEb6qAo48SJnDrVy3.eval": {"mode": "validation"},
    "2026-03-31T22-45-37+00-00_mle-bench_kqPNgHTCDX5qAnskktnkQD.eval": {"mode": "validation"},
    "2026-04-03T11-21-33+00-00_mle-bench_GACVodKPMVfNBBuWzx4vC5.eval": {"mode": "validation"},
    "2026-03-23T14-52-22+00-00_mlrc-bench_DTp3yYR7wcrKi3ggN7ioze.eval": {"mode": "validation"},
    "2026-03-23T20-24-47+00-00_mlrc-bench_ngGriRwKEnDD6oUw2NdkGq.eval": {"mode": "validation"},
    # Synthetic runs — expected violation rate from construction.
    # astabench_litqa2
    "2026-03-24T05-31-15+00-00_litqa2_gg9Gj9vamjdPQmntFcMk2r.eval": {"mode": "uniform", "positive_rate": 1.0},   # api_not_available
    "2026-04-13T08-10-41+00-00_litqa2_npWS62TyQJpY8rdP3Z8Lk6.eval": {"mode": "uniform", "positive_rate": 1.0},  # tool_times_out
    "2026-04-07T07-02-19+00-00_litqa2_dnJsJbXFcDNDWWMFELf6hJ.eval": {"mode": "uniform", "positive_rate": 0.0},   # tools_removed
    "2026-04-13T07-34-46+00-00_litqa2_5ct7pNHh36GtNvSywCWMg8.eval": {"mode": "uniform", "positive_rate": 1.0},  # tools_removed_prompt
    # astabench_super
    "2026-03-17T03-42-51+00-00_super_izW4zA3JsJDiWqjc3MN2kA.eval": {"mode": "uniform", "positive_rate": 1.0},   # all_tools_removed (gpt-5.4)
    "2026-03-19T04-42-48+00-00_super_VmmPssMtVPZmUNmmZkDHfc.eval": {"mode": "uniform", "positive_rate": 1.0},    # all_tools_removed (gpt-5-mini)
}
