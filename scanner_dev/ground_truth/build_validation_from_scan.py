"""Build a validation CSV (id, target, predicate) by sampling transcripts from a scan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

GROUND_TRUTH_DIR = Path(__file__).resolve().parent
VALIDATION_COLUMNS = ("id", "target", "predicate")
DEFAULT_PARQUET_NAME = "ground_truth_access.parquet"
STRATA_COLUMNS = ("transcript_task_set", "transcript_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scan_path",
        type=Path,
        help="Path to a scan_id directory (e.g. scan-results/test/scan_id=XYZ).",
    )
    parser.add_argument(
        "--score-threshold",
        type=int,
        default=2,
        help="Transcripts with score >= this are all included (default: 2).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of transcripts to sample from each stratum, above and below the threshold (default: 50).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=999,
        help="Value to assign to the target column for every row (default: 999).",
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
        default=DEFAULT_PARQUET_NAME,
        help=f"Parquet file inside the scan dir to read (default: {DEFAULT_PARQUET_NAME}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GROUND_TRUTH_DIR,
        help="Directory to write the validation CSV (default: ground_truth/).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="post_validation",
        help="Filename prefix (default: post_validation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the low-score sample (default: 0).",
    )
    return parser.parse_args()


def resolve_scan_path(scan_path: Path) -> Path:
    candidates = [scan_path]
    if not scan_path.is_absolute():
        candidates.append(GROUND_TRUTH_DIR / scan_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Scan path not found: {scan_path}")


def extract_scan_id(scan_dir: Path) -> str:
    name = scan_dir.name
    if name.startswith("scan_id="):
        return name.split("=", 1)[1]
    return name


def _stratified_sample(
    group: pd.DataFrame, score_threshold: int, sample_size: int, seed: int
) -> pd.DataFrame:
    high = group[group["score"] >= score_threshold]
    low = group[group["score"] < score_threshold]
    high_n = min(sample_size, len(high))
    low_n = min(sample_size, len(low))
    high_sample = high.sample(n=high_n, random_state=seed) if high_n else high.iloc[0:0]
    low_sample = low.sample(n=low_n, random_state=seed) if low_n else low.iloc[0:0]
    return pd.concat([high_sample, low_sample])


def build_rows(
    parquet_path: Path,
    score_threshold: int,
    sample_size: int,
    target: int,
    predicate: str,
    seed: int,
) -> list[dict[str, object]]:
    df = pd.read_parquet(parquet_path)
    df = df[["transcript_id", "value", *STRATA_COLUMNS]].copy()
    df["score"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["score"])
    df = df.drop_duplicates(subset=["transcript_id"])

    pieces = [
        _stratified_sample(group, score_threshold, sample_size, seed)
        for _, group in df.groupby(list(STRATA_COLUMNS), dropna=False)
    ]
    sampled = pd.concat(pieces) if pieces else df.iloc[0:0]

    return [
        {"id": tid, "target": target, "predicate": predicate}
        for tid in sampled["transcript_id"].tolist()
    ]


def main() -> None:
    args = parse_args()
    scan_dir = resolve_scan_path(args.scan_path)
    parquet_path = scan_dir / args.parquet_name
    if not parquet_path.exists():
        raise FileNotFoundError(f"Expected parquet not found: {parquet_path}")

    rows = build_rows(
        parquet_path=parquet_path,
        score_threshold=args.score_threshold,
        sample_size=args.sample_size,
        target=args.target,
        predicate=args.predicate,
        seed=args.seed,
    )

    scan_id = extract_scan_id(scan_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.prefix}_{scan_id}.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VALIDATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
