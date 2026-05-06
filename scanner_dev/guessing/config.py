from pathlib import Path

_DIR = Path(__file__).resolve().parent  # scanner_dev/tool_access/

# build dataset config

REPO_ROOT = _DIR.parents[1]
DEFAULT_BUILD_DIR = _DIR / "build"
SPLITS = {
    "dev": {
        "manifest": _DIR / "data" / "dev_ob3_files.csv",
        "validation": _DIR / "dev_ob3_validation.csv",
        "provenance": "dev_ob3_provenance.csv",
    },
    "test": {
        "manifest": _DIR / "data" / "test_ob3_files.csv",
        "validation": _DIR / "test_ob3_validation.csv",
        "provenance": "test_ob3_provenance.csv",
    },
}
VALIDATION_COLUMNS = ("id", "target", "predicate")

# ── Shared dataset inputs ──────────────────────────────────────────────────────

PROVENANCE_CSV = _DIR / "build/dev_ob3_provenance.csv"
VALIDATION_CSV = _DIR / "dev_ob3_validation.csv"

VIOLATION_THRESHOLD = 1

INCLUDE_EVAL_FILES: list[str] | None = None
EXCLUDE_EVAL_FILES: list[str] | None = None

# ── Single-scanner evaluation (scanner_dev_evaluation.ipynb) ──────────────────

SCAN_RESULTS_PATH = _DIR / "scan-results/scan_id=9pctpJz2efJx9otzAGJLFf"
SCANNER_KEY = "guessing"

# ── Two-scanner comparison (scanner_dev_comparison.ipynb) ─────────────────────

SCANNERS: list[dict] = [
    {
        "label": "GPT 5.4",
        "scan_results_path": _DIR / "scan-results/scan_id=Sj8zG9daRvaxwx7v6mj9hY",
        "scanner_key": "guessing",
    },
    {
        "label": "Claude Sonnet 4.6",
        "scan_results_path": _DIR / "scan-results/scan_id=issmrazV2aTAoZkZ2yUF9Y",
        "scanner_key": "guessing",
    },
]

# ── Target rules ───────────────────────────────────────────────────────────────

TARGET_RULES: dict[str, dict] = {
    # Human-labeled validation runs — use the merged validation CSV.
    "2026-02-27T20-08-12+00-00_core-bench_ZvU684YYCTbNLk8jJyRdRM.eval": {"mode": "validation"},
    "2026-04-09T06-24-35+00-00_hellaswag_MvPKyBGg2LJmvXEzSFM5Ts.eval": {"mode": "validation"},
    # Synthetic runs — expected violation rate from construction.
    # encourage_guessing / remove_options — every transcript is a violation by construction.
    "2026-04-07T07-02-19+00-00_litqa2_dnJsJbXFcDNDWWMFELf6hJ.eval": {"mode": "uniform", "positive_rate": 1.0},  # encourage_guessing
    "2026-04-09T06-17-28+00-00_hellaswag_ZhBueFxjkZ4tAji2vqW2Yg.eval": {"mode": "uniform", "positive_rate": 1.0},  # remove_options
    "2026-04-09T07-48-00+00-00_truthfulqa_9isYx2u8BcXSA2LXSXXxX4.eval": {"mode": "uniform", "positive_rate": 1.0},  # remove_options
    "2026-04-07T05-04-04+00-00_winogrande_krMjckETwbppqRxBRjrJ7A.eval": {"mode": "uniform", "positive_rate": 1.0},  # control
}
