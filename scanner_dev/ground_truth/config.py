from pathlib import Path

_DIR = Path(__file__).resolve().parent          # scanner_dev/ground_truth/
_SCANS_DIR = Path(__file__).parents[2] / "evals/scans/ground_truth_access/test/scan-results"

# build dataset config
REPO_ROOT = _DIR.parents[1]
DEFAULT_BUILD_DIR = _DIR / "build"
SPLITS = {
    "dev": {
        "manifest": _DIR / "data" / "dev_t5_files.csv",
        "validation": _DIR / "dev_t5_validation.csv",
        "provenance": "dev_t5_provenance.csv",
    },
    "test": {
        "manifest": _DIR / "data" / "test_t5_files.csv",
        "validation": _DIR / "test_t5_validation.csv",
        "provenance": "test_t5_provenance.csv",
    },
}
VALIDATION_COLUMNS = ("id", "target", "predicate")

# ── Shared dataset inputs ──────────────────────────────────────────────────────

PROVENANCE_CSV = _DIR / "build/dev_t5_provenance.csv"
VALIDATION_CSV = _DIR / "dev_t5_validation.csv"

VIOLATION_THRESHOLD = 2

INCLUDE_EVAL_FILES: list[str] | None = None
EXCLUDE_EVAL_FILES: list[str] | None = None

# ── Single-scanner evaluation (scanner_dev_evaluation.ipynb) ──────────────────

SCAN_RESULTS_PATH = _SCANS_DIR / "scan_id=MH6UxNTcbmXqXVqJnmmb7G"
SCANNER_KEY = "ground_truth_access"

# ── Two-scanner comparison (scanner_dev_comparison.ipynb) ─────────────────────

SCANNERS: list[dict] = [
    {
        "label": "GPT 5.4 r2",
        "scan_results_path": _SCANS_DIR / "scan_id=3hdEFtG2KsMd8BbxWPGLYp",
        "scanner_key": "ground_truth_access",
    },
    {
        "label": "GPT 5.4 r1",
        "scan_results_path": _SCANS_DIR / "scan_id=3bTfaerfMoM8URaYbfVEwX",
        "scanner_key": "ground_truth_access",
    },
]

# ── Target rules ───────────────────────────────────────────────────────────────

TARGET_RULES: dict[str, dict] = {
    # Human-labeled default runs — use the merged validation CSV.
    "2026-03-04T03-18-54+00-00_core-bench_6oon8Nk4txsRYbxWwDCtx7.eval": {"mode": "validation"},
    "2026-03-05T07-19-51+00-00_terminal-bench-2-0_EFZcEhAZEwRExw66VnVjVD.eval": {"mode": "validation"},
    "2026-03-16T18-33-04+00-00_swe-bench-verified-mini_dJnAEEf9EvTnvQkjVvuvZJ.eval": {"mode": "validation"},
    "2026-03-17T01-52-03+00-00_swe-bench_JeUnoWAJph9k8xDoDQ8ZqE.eval": {"mode": "validation"},
    "2026-03-21T09-34-42+00-00_mle-bench_awRHnzf67AWSZYtXs7q2bf.eval": {"mode": "validation"},
    "2026-03-31T22-45-37+00-00_mle-bench_kqPNgHTCDX5qAnskktnkQD.eval": {"mode": "validation"},
    # Synthetic t5-contamination / t5-web runs — every transcript is a violation by construction.
    "2026-03-27T14-34-51+00-00_swe-bench-verified-mini_n3CMjefQBLv9aioCrWJruf.eval": {"mode": "uniform", "positive_rate": 1.0},
    "2026-03-27T14-55-06+00-00_swe-bench-verified-mini_dLD6CXifonCqHfUTa5HVSw.eval": {"mode": "uniform", "positive_rate": 1.0},
    "2026-04-05T18-09-59+00-00_swe-bench_5GB8pn7xBw9aQXNXZqakmR.eval": {"mode": "uniform", "positive_rate": 1.0},
    "2026-04-05T18-27-27+00-00_swe-bench_38WCvSh5XQpJH98tnAKxit.eval": {"mode": "uniform", "positive_rate": 1.0},
    "2026-04-06T02-03-57+00-00_core-bench_Lcen3QviFFvtpzLR2M7igN.eval": {"mode": "uniform", "positive_rate": 1.0},
    "2026-04-06T02-20-13+00-00_core-bench_7Nd6ZmaA5wszv8cSbeVrzY.eval": {"mode": "uniform", "positive_rate": 1.0},
    "2026-04-06T08-43-14+00-00_terminal-bench-2-0-oracle_7LhUD5D6mWs5Ygr2L7rC4k.eval": {"mode": "uniform", "positive_rate": 1.0},
    "2026-04-06T23-30-27+00-00_terminal-bench-2-0-oracle_ETAznfha6MQiFnGWFAGDdY.eval": {"mode": "uniform", "positive_rate": 1.0},
    "2026-04-07T01-59-51+00-00_terminal-bench-2-0-oracle_3Dn4fjyyh3FRdsmHzAqu9Q.eval": {"mode": "uniform", "positive_rate": 1.0},
}
