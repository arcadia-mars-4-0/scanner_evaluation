"""Utilities for loading eval logs, scan results, and analysis tables."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd


TRANSCRIPT_METADATA_COLUMNS = [
    "transcript_id",
    "task_id",
    "benchmark",
    "eval_subset",
    "eval_file",
    "model",
    "transcript_score",
    "transcript_success",
]


def load_eval_logs(
    eval_logs_dir: str | Path,
    score_key: str | None = None,
    success_fn: Callable | None = None,
    exclude_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Load transcript metadata from ``.eval`` log files."""
    eval_logs_dir = Path(eval_logs_dir)
    exclude_patterns = exclude_patterns or []
    eval_files = sorted(eval_logs_dir.rglob("*.eval"))
    if exclude_patterns:
        eval_files = [
            f for f in eval_files
            if not any(pat in str(f) for pat in exclude_patterns)
        ]
    if not eval_files:
        raise FileNotFoundError(f"No .eval files found under {eval_logs_dir}")

    check_success = success_fn or _is_success

    rows: list[dict] = []
    for eval_file in eval_files:
        subset = _subset_from_path(eval_file, eval_logs_dir)
        with zipfile.ZipFile(eval_file) as zf:
            header = json.loads(zf.read("header.json"))
            model = header.get("eval", {}).get("model")
            benchmark = header.get("eval", {}).get("task")
            summaries = json.loads(zf.read("summaries.json"))
            for summary in summaries:
                scores = summary.get("scores", {})
                score_val = None
                if scores:
                    first_score = next(iter(scores.values()))
                    score_val = first_score.get("value")
                if score_key is not None and isinstance(score_val, dict):
                    score_val = score_val.get(score_key)
                rows.append(
                    {
                        "transcript_id": summary["uuid"],
                        "task_id": summary["id"],
                        "transcript_score": score_val,
                        "transcript_success": check_success(score_val),
                        "eval_file": str(eval_file.relative_to(eval_logs_dir)),
                        "eval_subset": subset,
                        "model": model,
                        "benchmark": benchmark,
                    }
                )

    return pd.DataFrame(rows)


def load_scan_results(
    scan_results_dir: str | Path,
    top_level_only: bool = False,
) -> pd.DataFrame:
    """Load scanner output parquet files from scan-result directories."""
    scan_results_dir = Path(scan_results_dir)
    if scan_results_dir.name.startswith("scan_id=") and scan_results_dir.is_dir():
        scan_dirs = [scan_results_dir]
    elif top_level_only:
        scan_dirs = sorted(scan_results_dir.glob("scan_id=*"))
    else:
        scan_dirs = sorted(scan_results_dir.rglob("scan_id=*"))
    if not scan_dirs:
        raise FileNotFoundError(
            f"No scan_id=* directories found under {scan_results_dir}"
        )

    frames: list[pd.DataFrame] = []
    for scan_dir in scan_dirs:
        parquet_files = sorted(scan_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        scan_meta_path = scan_dir / "_scan.json"
        scanner_model = None
        benchmark = None
        scan_name = None
        scan_timestamp = None
        if scan_meta_path.exists():
            with open(scan_meta_path) as f:
                scan_meta = json.load(f)
            scanner_model = scan_meta.get("model", {}).get("model")
            benchmark = scan_meta.get("scan_name")
            scan_name = scan_meta.get("scan_name")
            scan_timestamp = scan_meta.get("timestamp")

        scanner_source = str(scan_dir.relative_to(scan_results_dir))

        for pq_path in parquet_files:
            df = pd.read_parquet(pq_path)
            df["scanner_model"] = scanner_model
            df["benchmark"] = benchmark
            df["scan_name"] = scan_name
            df["scan_timestamp"] = scan_timestamp
            df["scanner_source"] = scanner_source
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["value_num"] = pd.to_numeric(combined["value"], errors="coerce")
    return combined


def load_validations(
    validation_dir: str | Path,
    prefix: str = "swe_bench_",
) -> pd.DataFrame:
    """Load validation CSVs and return a single DataFrame keyed by transcript ID."""
    validation_dir = Path(validation_dir)
    csv_paths = sorted(validation_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {validation_dir}")

    result: pd.DataFrame | None = None
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "id" not in df.columns or "target" not in df.columns:
            continue
        col_name = csv_path.stem
        if col_name.startswith(prefix):
            col_name = col_name[len(prefix) :]
        validation = df[["id", "target"]].rename(
            columns={"id": "transcript_id", "target": col_name}
        )
        if result is None:
            result = validation
        else:
            result = result.merge(validation, on="transcript_id", how="outer")
    if result is None:
        raise FileNotFoundError(
            f"No CSV files with 'id' and 'target' columns found under {validation_dir}"
        )
    return result


def load_eval_runs_csv(eval_runs_csv: str | Path) -> pd.DataFrame:
    """Load preprocessed eval metadata from CSV."""
    eval_runs = pd.read_csv(eval_runs_csv)
    expected = {"transcript_id", "benchmark", "eval_subset"}
    missing = expected - set(eval_runs.columns)
    if missing:
        raise ValueError(
            f"Eval runs CSV is missing required columns: {sorted(missing)}"
        )
    return eval_runs


def load_human_labels_wide(human_labels_wide_csv: str | Path) -> pd.DataFrame:
    """Load wide human label CSV."""
    labels = pd.read_csv(human_labels_wide_csv)
    expected = {"transcript_id", "benchmark"}
    missing = expected - set(labels.columns)
    if missing:
        raise ValueError(
            f"Human labels wide CSV is missing required columns: {sorted(missing)}"
        )
    return labels


def load_human_labels_long(human_labels_long_csv: str | Path) -> pd.DataFrame:
    """Load long human label CSV."""
    labels = pd.read_csv(human_labels_long_csv)
    expected = {
        "transcript_id",
        "benchmark",
        "label_name",
        "criteria",
        "labeler",
        "target",
    }
    missing = expected - set(labels.columns)
    if missing:
        raise ValueError(
            f"Human labels long CSV is missing required columns: {sorted(missing)}"
        )
    labels["target_num"] = pd.to_numeric(labels["target"], errors="coerce")
    return labels


def build_summary(
    logs: pd.DataFrame,
    scans: pd.DataFrame | None = None,
    validation_dir: str | Path | None = None,
    validation_prefix: str = "swe_bench_",
) -> pd.DataFrame:
    """Build a one-row-per-transcript summary from eval logs, scans, and validations."""
    summary = logs.copy()

    if scans is not None:
        agg = (
            scans.groupby(["transcript_id", "scanner_key"], dropna=False)
            .agg(value=("value_num", "first"))
            .reset_index()
        )
        pivot = agg.pivot_table(
            index="transcript_id",
            columns="scanner_key",
            values="value",
            aggfunc="first",
        ).reset_index()
        pivot.columns.name = None
        summary = summary.merge(pivot, on="transcript_id", how="left")

    if validation_dir is not None:
        validations = load_validations(validation_dir, prefix=validation_prefix)
        summary = summary.merge(validations, on="transcript_id", how="left")

    return summary


def build_analysis_frame(
    scans: pd.DataFrame,
    eval_runs: pd.DataFrame,
    human_labels_long: pd.DataFrame | None = None,
    include_scanner_sources: Iterable[str] | None = None,
    exclude_scanner_sources: Iterable[str] | None = None,
    include_subsets: Iterable[str] | None = None,
    exclude_subsets: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build a normalized scan-analysis table."""
    analysis = scans.copy()
    analysis = analysis.merge(
        eval_runs[TRANSCRIPT_METADATA_COLUMNS].drop_duplicates(
            subset=["transcript_id", "benchmark"]
        ),
        on=["transcript_id", "benchmark"],
        how="left",
    )

    analysis = _filter_values(
        analysis,
        "scanner_source",
        include_values=include_scanner_sources,
        exclude_values=exclude_scanner_sources,
    )
    analysis = _filter_values(
        analysis,
        "eval_subset",
        include_values=include_subsets,
        exclude_values=exclude_subsets,
    )

    analysis["has_eval_metadata"] = analysis["eval_subset"].notna()
    analysis["value_num"] = pd.to_numeric(analysis["value_num"], errors="coerce")

    if human_labels_long is not None:
        coverage = (
            human_labels_long.groupby(["transcript_id", "benchmark"], dropna=False)
            .agg(
                human_label_count=("label_name", "nunique"),
                human_label_names=("label_name", lambda x: sorted(set(x))),
            )
            .reset_index()
        )
        analysis = analysis.merge(
            coverage,
            on=["transcript_id", "benchmark"],
            how="left",
        )
        analysis["has_human_labels"] = analysis["human_label_count"].fillna(0).gt(0)
    else:
        analysis["human_label_count"] = 0
        analysis["human_label_names"] = [[] for _ in range(len(analysis))]
        analysis["has_human_labels"] = False

    return analysis


def build_comparison_table(
    analysis: pd.DataFrame,
    human_labels_wide: pd.DataFrame | None = None,
    human_label_names: Iterable[str] | None = None,
    include_scanner_source: bool = True,
) -> pd.DataFrame:
    """Pivot scanner grades into one row per transcript or transcript x source."""
    index_cols = ["transcript_id"]
    if include_scanner_source and "scanner_source" in analysis.columns:
        source_count = analysis["scanner_source"].nunique(dropna=True)
        if source_count > 1:
            index_cols.append("scanner_source")

    meta_cols = [
        col
        for col in TRANSCRIPT_METADATA_COLUMNS
        if col in analysis.columns and col not in index_cols
    ]
    if "scanner_model" in analysis.columns:
        meta_cols.append("scanner_model")

    meta = analysis[index_cols + meta_cols].drop_duplicates(subset=index_cols)

    scanner_values = (
        analysis.groupby(index_cols + ["scanner_key"], dropna=False)
        .agg(scanner_grade=("value_num", "first"))
        .reset_index()
    )
    wide = scanner_values.pivot_table(
        index=index_cols,
        columns="scanner_key",
        values="scanner_grade",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    comparison = meta.merge(wide, on=index_cols, how="left")

    if human_labels_wide is not None:
        human_cols = ["transcript_id", "benchmark"]
        available = set(human_labels_wide.columns)
        if human_label_names is None:
            selected_labels = [
                col for col in human_labels_wide.columns if col not in human_cols
            ]
        else:
            selected_labels = [col for col in human_label_names if col in available]
        comparison = comparison.merge(
            human_labels_wide[human_cols + selected_labels],
            on=["transcript_id", "benchmark"],
            how="left",
        )

    return comparison


def build_human_analysis_frame(
    eval_runs: pd.DataFrame,
    human_labels_long: pd.DataFrame,
    include_subsets: Iterable[str] | None = None,
    exclude_subsets: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Attach eval metadata to long-form human labels for plotting and metrics."""
    human = human_labels_long.merge(
        eval_runs[TRANSCRIPT_METADATA_COLUMNS].drop_duplicates(
            subset=["transcript_id", "benchmark"]
        ),
        on=["transcript_id", "benchmark"],
        how="left",
    )
    human = _filter_values(
        human,
        "eval_subset",
        include_values=include_subsets,
        exclude_values=exclude_subsets,
    )
    human["target_num"] = pd.to_numeric(human["target_num"], errors="coerce")
    return human


def add_violation_flags(
    frame: pd.DataFrame,
    threshold: float = 2,
    *,
    score_col: str = "value_num",
    output_col: str = "is_violation",
) -> pd.DataFrame:
    """Add a binary violation flag to a DataFrame."""
    flagged = frame.copy()
    numeric_scores = pd.to_numeric(flagged[score_col], errors="coerce")
    flagged[output_col] = pd.Series(
        np.where(numeric_scores.isna(), pd.NA, numeric_scores.ge(threshold)),
        index=flagged.index,
        dtype="boolean",
    )
    return flagged


def summarize_coverage(
    analysis: pd.DataFrame,
    human_labels_wide: pd.DataFrame | None = None,
    human_label_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Summarize scan/eval/human coverage for the current selection."""
    rows: list[dict] = [
        {
            "metric": "scan_rows",
            "value": int(len(analysis)),
        },
        {
            "metric": "unique_transcripts_in_scans",
            "value": int(analysis["transcript_id"].nunique()),
        },
        {
            "metric": "scanner_sources",
            "value": int(analysis["scanner_source"].nunique()),
        },
        {
            "metric": "scanners",
            "value": int(analysis["scanner_key"].nunique()),
        },
        {
            "metric": "rows_with_eval_metadata",
            "value": int(analysis["has_eval_metadata"].sum()),
        },
        {
            "metric": "transcripts_with_eval_metadata",
            "value": int(
                analysis.loc[analysis["has_eval_metadata"], "transcript_id"].nunique()
            ),
        },
    ]

    if human_labels_wide is not None:
        comparison = build_comparison_table(
            analysis,
            human_labels_wide=human_labels_wide,
            human_label_names=human_label_names,
        )
        human_cols = [
            col
            for col in comparison.columns
            if col not in set(TRANSCRIPT_METADATA_COLUMNS + ["scanner_source", "scanner_model"])
            and col not in set(analysis["scanner_key"].dropna().unique())
        ]
        for col in human_cols:
            rows.append(
                {
                    "metric": f"human_label_coverage::{col}",
                    "value": int(comparison[col].notna().sum()),
                }
            )

    return pd.DataFrame(rows)


def summarize_grade_distribution(
    analysis: pd.DataFrame,
    *,
    group_cols: list[str],
    grade_col: str = "value_num",
) -> pd.DataFrame:
    """Summarize grade counts and proportions by group."""
    counts = (
        analysis.dropna(subset=[grade_col])
        .assign(grade=lambda df: pd.to_numeric(df[grade_col], errors="coerce"))
        .dropna(subset=["grade"])
        .assign(grade=lambda df: df["grade"].astype(int))
        .groupby(group_cols + ["grade"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(group_cols, dropna=False)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    return counts


def summarize_violation_rates(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    flag_col: str = "is_violation",
) -> pd.DataFrame:
    """Summarize violation rates by group."""
    valid = frame[frame[flag_col].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=group_cols + ["violation_rate", "n"])
    summary = (
        valid.groupby(group_cols, dropna=False)[flag_col]
        .agg(violation_rate="mean", n="size")
        .reset_index()
    )
    return summary


def materialize_targets(
    comparison: pd.DataFrame,
    target_rules: dict,
    *,
    scanner_key: str,
    violation_threshold: float = 2,
) -> pd.DataFrame:
    """Create transcript-level targets for one scanner based on subset rules."""
    if scanner_key not in comparison.columns:
        raise KeyError(f"Scanner column not found in comparison table: {scanner_key}")

    target_frame = comparison.copy()
    prediction_scores = pd.to_numeric(target_frame[scanner_key], errors="coerce")
    target_frame["prediction"] = pd.Series(
        np.where(
            prediction_scores.isna(),
            pd.NA,
            prediction_scores.ge(violation_threshold),
        ),
        index=target_frame.index,
        dtype="boolean",
    )
    target_frame["target_source"] = pd.NA
    target_frame["target_label_name"] = pd.NA
    target_frame["target"] = pd.Series(pd.NA, index=target_frame.index, dtype="boolean")

    for subset, rule in target_rules.items():
        subset_mask = target_frame["eval_subset"] == subset
        if not subset_mask.any():
            continue
        resolved_rule = _resolve_target_rule(rule, scanner_key)
        mode = resolved_rule.get("mode")

        if mode == "human":
            label_name = resolved_rule["label_name"]
            if label_name not in target_frame.columns:
                raise KeyError(
                    f"Target rule for subset '{subset}' references missing human label "
                    f"column '{label_name}'"
                )
            labels = pd.to_numeric(target_frame.loc[subset_mask, label_name], errors="coerce")
            target_frame.loc[subset_mask, "target"] = pd.Series(
                np.where(labels.isna(), pd.NA, labels.ge(violation_threshold)),
                index=target_frame.loc[subset_mask].index,
                dtype="boolean",
            )
            target_frame.loc[subset_mask, "target_source"] = "human"
            target_frame.loc[subset_mask, "target_label_name"] = label_name
        elif mode == "uniform":
            positive_rate = resolved_rule.get("positive_rate")
            if positive_rate not in {0, 0.0, 1, 1.0}:
                raise ValueError(
                    f"Uniform target rules only support 0.0 or 1.0 positive_rate. "
                    f"Got {positive_rate!r} for subset '{subset}'."
                )
            target_frame.loc[subset_mask, "target"] = bool(positive_rate)
            target_frame.loc[subset_mask, "target_source"] = "uniform"
            target_frame.loc[subset_mask, "target_label_name"] = f"uniform::{positive_rate}"
        else:
            raise ValueError(f"Unsupported target rule mode for subset '{subset}': {mode!r}")

    return target_frame


def summarize_performance_metrics(
    comparison: pd.DataFrame,
    target_rules: dict,
    *,
    scanner_keys: Iterable[str],
    violation_threshold: float = 2,
) -> pd.DataFrame:
    """Compute accuracy, sensitivity, and specificity by subset and scanner."""
    metrics_frames: list[pd.DataFrame] = []
    source_count = comparison["scanner_source"].nunique() if "scanner_source" in comparison.columns else 0

    for scanner_key in scanner_keys:
        targeted = materialize_targets(
            comparison,
            target_rules,
            scanner_key=scanner_key,
            violation_threshold=violation_threshold,
        )
        valid = targeted[targeted["target"].notna() & targeted["prediction"].notna()].copy()
        if valid.empty:
            continue

        group_cols = ["benchmark", "eval_subset"]
        if "scanner_source" in valid.columns and source_count > 1:
            group_cols.append("scanner_source")

        for group_values, group_df in valid.groupby(group_cols, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            group_map = dict(zip(group_cols, group_values, strict=False))

            prediction = group_df["prediction"].astype(bool)
            target = group_df["target"].astype(bool)

            tp = int((prediction & target).sum())
            tn = int((~prediction & ~target).sum())
            fp = int((prediction & ~target).sum())
            fn = int((~prediction & target).sum())

            pos_denom = tp + fn
            neg_denom = tn + fp
            total = len(group_df)

            metrics_frames.append(
                pd.DataFrame(
                    [
                        {
                            **group_map,
                            "scanner_key": scanner_key,
                            "accuracy": (tp + tn) / total if total else np.nan,
                            "sensitivity": tp / pos_denom if pos_denom else np.nan,
                            "specificity": tn / neg_denom if neg_denom else np.nan,
                            "tp": tp,
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                            "n": int(total),
                            "n_missing_target": int(targeted["target"].isna().sum()),
                            "target_source": _single_value(group_df["target_source"]),
                            "target_label_name": _single_value(
                                group_df["target_label_name"]
                            ),
                        }
                    ]
                )
            )

    if not metrics_frames:
        return pd.DataFrame(
            columns=[
                "benchmark",
                "eval_subset",
                "scanner_key",
                "accuracy",
                "sensitivity",
                "specificity",
                "tp",
                "tn",
                "fp",
                "fn",
                "n",
                "n_missing_target",
                "target_source",
                "target_label_name",
            ]
        )

    return pd.concat(metrics_frames, ignore_index=True)


def summarize_scanner_comparison(
    comparison: pd.DataFrame,
    *,
    scanner_keys: Iterable[str],
    violation_threshold: float = 2,
) -> pd.DataFrame:
    """Build a compact subset-level scanner comparison table."""
    rows: list[dict] = []
    include_source = (
        "scanner_source" in comparison.columns
        and comparison["scanner_source"].nunique(dropna=True) > 1
    )
    group_cols = ["benchmark", "eval_subset"]
    if include_source:
        group_cols.append("scanner_source")

    for scanner_key in scanner_keys:
        if scanner_key not in comparison.columns:
            continue
        work = comparison[group_cols + [scanner_key]].copy()
        work["scanner_grade"] = pd.to_numeric(work[scanner_key], errors="coerce")
        work["is_violation"] = work["scanner_grade"].ge(violation_threshold)
        grouped = (
            work.dropna(subset=["scanner_grade"])
            .groupby(group_cols, dropna=False)
            .agg(
                mean_grade=("scanner_grade", "mean"),
                violation_rate=("is_violation", "mean"),
                n=("scanner_grade", "size"),
            )
            .reset_index()
        )
        grouped["scanner_key"] = scanner_key
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(
            columns=["benchmark", "eval_subset", "scanner_key", "mean_grade", "violation_rate", "n"]
        )
    return pd.concat(rows, ignore_index=True)


def format_metric_percent(metric_frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a formatted copy of a metrics frame."""
    formatted = metric_frame.copy()
    for col in columns:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "NA"
            )
    return formatted


def _filter_values(
    frame: pd.DataFrame,
    column: str,
    *,
    include_values: Iterable[str] | None = None,
    exclude_values: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filter a frame by include/exclude value lists."""
    filtered = frame.copy()
    if include_values:
        include_values = set(include_values)
        filtered = filtered[filtered[column].isin(include_values)]
    if exclude_values:
        exclude_values = set(exclude_values)
        filtered = filtered[~filtered[column].isin(exclude_values)]
    return filtered


def _resolve_target_rule(rule: dict, scanner_key: str) -> dict:
    """Resolve a target rule, allowing scanner-specific overrides."""
    scanner_overrides = rule.get("scanner_overrides", {})
    if scanner_key in scanner_overrides:
        resolved = dict(rule)
        resolved.update(scanner_overrides[scanner_key])
        resolved.pop("scanner_overrides", None)
        return resolved
    return dict(rule)


def _single_value(series: pd.Series):
    """Return the single non-null value in a Series, or the first when mixed."""
    values = [value for value in series.dropna().unique().tolist()]
    if not values:
        return pd.NA
    return values[0]


def _subset_from_path(eval_file: Path, root: Path) -> str:
    """Derive an eval subset label from the file's position in the directory tree."""
    parent = str(eval_file.relative_to(root).parent)
    return "default" if parent == "." else parent


def _is_success(score_value) -> bool:
    """Determine whether a score value represents success."""
    if score_value is None:
        return False
    if isinstance(score_value, str):
        return score_value == "C"
    return float(score_value) > 0
