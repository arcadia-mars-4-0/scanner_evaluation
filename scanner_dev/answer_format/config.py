from pathlib import Path

_DIR = Path(__file__).resolve().parent  # scanner_dev/tool_access/

# build dataset config

REPO_ROOT = _DIR.parents[1]
DEFAULT_BUILD_DIR = _DIR / "build"
SPLITS = {
    "dev": {
        "manifest": _DIR / "data" / "dev_oh1_files.csv",
        "validation": _DIR / "dev_oh1_validation.csv",
        "provenance": "dev_oh1_provenance.csv",
    },
    "test": {
        "manifest": _DIR / "data" / "test_oh1_files.csv",
        "validation": _DIR / "test_oh1_validation.csv",
        "provenance": "test_oh1_provenance.csv",
    },
}
VALIDATION_COLUMNS = ("id", "target", "predicate")

# ── Shared dataset inputs ──────────────────────────────────────────────────────

PROVENANCE_CSV = _DIR / "build/dev_oh1_provenance.csv"
VALIDATION_CSV = _DIR / "dev_oh1_validation.csv"

VIOLATION_THRESHOLD = 1

INCLUDE_EVAL_FILES: list[str] | None = None
EXCLUDE_EVAL_FILES: list[str] | None = None

# ── Single-scanner evaluation (scanner_dev_evaluation.ipynb) ──────────────────

SCAN_RESULTS_PATH = _DIR / "scan-results/scan_id=oSn6KR5hvUvP96A8Ab9MDp"
SCANNER_KEY = "answer_format"

# ── Two-scanner comparison (scanner_dev_comparison.ipynb) ─────────────────────

SCANNERS: list[dict] = [
    {
        "label": "GPT 5.4",
        "scan_results_path": _DIR / "scan-results/scan_id=SrBWbcsKwN3MDTL7nzDtYG",
        "scanner_key": "answer_format",
    },
    {
        "label": "Claude Sonnet 4.6",
        "scan_results_path": _DIR / "scan-results/scan_id=i4oTKgzfKNb3nqqjd5BoDu",
        "scanner_key": "answer_format",
    },
]

# ── Target rules ───────────────────────────────────────────────────────────────

TARGET_RULES: dict[str, dict] = {
    # Human-labeled validation runs — use the merged validation CSV.
    "2026-02-27T20-08-12+00-00_core-bench_ZvU684YYCTbNLk8jJyRdRM.eval": {"mode": "validation"},
    "2026-03-05T07-19-51+00-00_terminal-bench-2-0_EFZcEhAZEwRExw66VnVjVD.eval": {"mode": "validation"},
    "2026-03-16T18-33-04+00-00_swe-bench-verified-mini_dJnAEEf9EvTnvQkjVvuvZJ.eval": {"mode": "validation"},
    "2026-03-17T01-52-03+00-00_swe-bench_JeUnoWAJph9k8xDoDQ8ZqE.eval": {"mode": "validation"},
    # Synthetic runs — expected violation rate from construction.
    # compute_eval
    "2026-04-14T01-48-00+00-00_compute-eval_MgGDQdWb86Ev9stJyiezGT.eval": {"mode": "uniform", "positive_rate": 1.0},  # obvious_failure_2
}
