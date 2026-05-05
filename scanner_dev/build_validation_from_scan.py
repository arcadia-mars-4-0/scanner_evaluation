"""Build a validation CSV (id, target, predicate) by sampling transcripts from a scan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

VALIDATION_COLUMNS = ("id", "target", "predicate")
TASK_SET_COLUMN = "transcript_task_set"
MODEL_COLUMN = "transcript_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scan_path",
        type=Path,
        help="Path to a scan_id directory (e.g. scanner_dev/guessing/scan-results/scan_id=XYZ).",
    )
    parser.add_argument(
        "--score-threshold",
        type=int,
        default=2,
        help="Transcripts with score >= this are flagged (default: 2).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Number of transcripts to sample from each stratum, above and below the threshold (default: 25).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=999,
        help="Placeholder value for the target column — fill in manually after export (default: 999).",
    )
    parser.add_argument(
        "--predicate",
        type=str,
        default="eq",
        help="Value to assign to the predicate column (default: eq).",
    )
    parser.add_argument(
        "--parquet-name",
        type=str,
        default=None,
        help="Parquet file inside the scan dir to read (auto-detected if not specified).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write the validation CSV "
            "(defaults to the scanner root, two levels above the scan_id dir)."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="post_validation_sample",
        help="Filename prefix for the output CSV (default: post_validation_sample).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling low-score transcripts (default: 0).",
    )
    parser.add_argument(
        "--stratify-by-model",
        action="store_true",
        help=(
            "Also stratify sampling by transcript_model. "
            "Default: pool transcripts across models so only task_set is the stratum."
        ),
    )
    return parser.parse_args()


def resolve_scan_path(scan_path: Path) -> Path:
    if scan_path.exists():
        return scan_path.resolve()
    resolved = Path.cwd() / scan_path
    if resolved.exists():
        return resolved.resolve()
    raise FileNotFoundError(f"Scan path not found: {scan_path}")


def detect_parquet(scan_dir: Path) -> Path:
    parquets = [p for p in scan_dir.glob("*.parquet") if not p.name.startswith("_")]
    if not parquets:
        raise FileNotFoundError(f"No parquet file found in {scan_dir}")
    if len(parquets) > 1:
        names = ", ".join(p.name for p in sorted(parquets))
        raise ValueError(
            f"Multiple parquet files in {scan_dir}: {names}. Use --parquet-name to specify one."
        )
    return parquets[0]


def infer_output_dir(scan_dir: Path) -> Path:
    # scan_dir is typically <scanner_root>/scan-results/scan_id=XYZ
    return scan_dir.parent.parent


def extract_scan_id(scan_dir: Path) -> str:
    name = scan_dir.name
    return name.split("=", 1)[1] if name.startswith("scan_id=") else name


def _stratified_sample(
    group: pd.DataFrame, score_threshold: int, sample_size: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    high = group[group["score"] >= score_threshold]
    low = group[group["score"] < score_threshold]
    high_sample = high.sample(n=min(sample_size, len(high)), random_state=seed) if len(high) else high.iloc[0:0]
    low_sample = low.sample(n=min(sample_size, len(low)), random_state=seed) if len(low) else low.iloc[0:0]
    return high_sample, low_sample


def build_rows(
    parquet_path: Path,
    score_threshold: int,
    sample_size: int,
    target: int,
    predicate: str,
    seed: int,
    stratify_by_model: bool = False,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    df = pd.read_parquet(parquet_path)
    candidate_strata = [TASK_SET_COLUMN]
    if stratify_by_model:
        candidate_strata.append(MODEL_COLUMN)
    available_strata = [c for c in candidate_strata if c in df.columns]
    df = df[["transcript_id", "value", *available_strata]].copy()
    df["score"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["score"])
    df = df.drop_duplicates(subset=["transcript_id"])

    high_pieces: list[pd.DataFrame] = []
    low_pieces: list[pd.DataFrame] = []
    report_rows: list[dict[str, object]] = []

    if available_strata:
        iter_groups = df.groupby(available_strata, dropna=False)
    else:
        iter_groups = [((), df)]

    for keys, group in iter_groups:
        if available_strata:
            key_values = (keys,) if len(available_strata) == 1 else tuple(keys)
        else:
            key_values = ()
        high_sample, low_sample = _stratified_sample(group, score_threshold, sample_size, seed)
        high_pieces.append(high_sample)
        low_pieces.append(low_sample)
        report_rows.append(
            {
                **dict(zip(available_strata, key_values)),
                "flagged_sampled": len(high_sample),
                "flagged_available": int((group["score"] >= score_threshold).sum()),
                "unflagged_sampled": len(low_sample),
                "unflagged_available": int((group["score"] < score_threshold).sum()),
                "total_sampled": len(high_sample) + len(low_sample),
            }
        )

    sampled = pd.concat(high_pieces + low_pieces) if (high_pieces or low_pieces) else df.iloc[0:0]
    rows = [
        {"id": tid, "target": target, "predicate": predicate}
        for tid in sampled["transcript_id"].tolist()
    ]
    return rows, pd.DataFrame(report_rows)


def print_report(report: pd.DataFrame) -> None:
    if report.empty:
        print("No strata found; report is empty.")
        return
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print("\nSample breakdown per stratum (flagged = score >= threshold):")
        print(report.to_string(index=False))
        stratum_cols = {TASK_SET_COLUMN, MODEL_COLUMN}
        totals = {col: int(report[col].sum()) for col in report.columns if col not in stratum_cols}
        print(f"\nTotals: {totals}")


def main() -> None:
    args = parse_args()
    scan_dir = resolve_scan_path(args.scan_path)

    parquet_path = scan_dir / args.parquet_name if args.parquet_name else detect_parquet(scan_dir)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    output_dir = args.output_dir or infer_output_dir(scan_dir)
    rows, report = build_rows(
        parquet_path=parquet_path,
        score_threshold=args.score_threshold,
        sample_size=args.sample_size,
        target=args.target,
        predicate=args.predicate,
        seed=args.seed,
        stratify_by_model=args.stratify_by_model,
    )

    scan_id = extract_scan_id(scan_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.prefix}_{scan_id}.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VALIDATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print_report(report)


if __name__ == "__main__":
    main()