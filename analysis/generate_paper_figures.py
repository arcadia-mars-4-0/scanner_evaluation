"""Generate scanner paper figures from a YAML config.

This script consolidates the plotting workflows from:

- analysis/scanner_validation.ipynb
- analysis/scanner_combined_results.ipynb
- analysis/scanner_combined_results_unweighted.ipynb
- analysis/scanner_paper_figures.ipynb

Example:
    python analysis/generate_paper_figures.py \
        --config analysis/configs/combined_test_nonT5.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scanner_evaluation_matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analysis_utils import (
    BENCHMARK_ALIASES,
    GRADE_LEVELS,
    SCORE_COLORS,
    bootstrap_kappa,
    bootstrap_threshold_metrics,
    bootstrap_weighted_metrics,
    confusion_matrix,
    draw_cm,
    format_metric_ci_columns,
    format_numeric_columns,
    make_savers,
    quadratic_weighted_kappa,
    shorten_model,
    stratified_ipw_weights,
    threshold_metrics,
    violation_rate,
    weighted_confusion_matrix,
    weighted_threshold_metrics,
)
from analysis.scan_utils import load_validations

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - pyarrow is already required by pandas parquet IO here.
    pq = None


CI_METRIC_COLS = [
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "quadratic_weighted_kappa",
]

COMPOSITES = [
    ("floor(mean)", "composite_floor_mean"),
    ("ceil(mean)", "composite_ceil_mean"),
    ("max", "composite_max"),
]

PREFERRED_SCANNER_ORDER = [
    "ground_truth_access",
    "tool_failure",
    "answer_format",
    "guessing",
]
PREFERRED_MODEL_ORDER = ["sonnet-4.6", "gpt-5.4"]

DISPLAY_LABELS = {
    "answer_format": "Answer Format",
    "ground_truth_access": "Ground Truth Access",
    "guessing": "Guessing",
    "tool_failure": "Tool Failure",
    "core_bench": "CORE-Bench",
    "cvebench": "CVE-Bench",
    "gpqa_diamond": "GPQA-Diamond",
    "hle": "HLE",
    "kernelbench": "KernelBench",
    "swe_bench": "SWE-Bench-Verified",
    "tau2_airline": "Tau2-Airline",
    "tau2_retail": "Tau2-Retail",
}

SCAN_RESULT_COLUMNS = [
    "transcript_id",
    "transcript_source_uri",
    "transcript_task_set",
    "transcript_model",
    "transcript_score",
    "transcript_success",
    "scanner_key",
    "value",
]


def display_label(name: str) -> str:
    return DISPLAY_LABELS.get(name, name)


def path_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ordered_values(values: pd.Series, preferred: list[str]) -> list[str]:
    present = [v for v in values.dropna().unique().tolist()]
    return sorted(
        present,
        key=lambda v: preferred.index(v) if v in preferred else len(preferred),
    )


def load_scan_results_for_figures(scan_results_dir: str | Path) -> pd.DataFrame:
    """Load only the scan-result columns required for plotting.

    The general-purpose loader in scan_utils reads large transcript/input/event
    columns. Those are useful for audit workflows but can push WSL over its
    memory limit when generating all figures.
    """
    scan_results_dir = Path(scan_results_dir)
    if scan_results_dir.name.startswith("scan_id=") and scan_results_dir.is_dir():
        scan_dirs = [scan_results_dir]
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
        for parquet_path in parquet_files:
            if pq is None:
                df = pd.read_parquet(parquet_path)
                df = df.reindex(columns=SCAN_RESULT_COLUMNS)
            else:
                available = set(pq.ParquetFile(parquet_path).schema.names)
                selected = [c for c in SCAN_RESULT_COLUMNS if c in available]
                df = pd.read_parquet(parquet_path, columns=selected)
                for col in SCAN_RESULT_COLUMNS:
                    if col not in df.columns:
                        df[col] = pd.NA
                df = df[SCAN_RESULT_COLUMNS]
            df["scanner_model"] = scanner_model
            df["benchmark"] = benchmark
            df["scan_name"] = scan_name
            df["scan_timestamp"] = scan_timestamp
            df["scanner_source"] = scanner_source
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No parquet files found under {scan_results_dir}")
    combined = pd.concat(frames, ignore_index=True)
    combined["value_num"] = pd.to_numeric(combined["value"], errors="coerce")
    return combined


def normalize_validation_files(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def normalize_scanner_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept both combined-notebook and scanner_validation config schemas."""
    if cfg.get("scanners"):
        return [dict(s) for s in cfg["scanners"]]

    required = {"target_scanner", "split"}
    if not required.issubset(cfg):
        raise ValueError("Config must contain `scanners` or single-scanner keys.")

    include_scan_ids = list(cfg.get("include_scan_ids") or [])
    exclude_scan_ids = set(cfg.get("exclude_scan_ids") or [])
    scan_ids = [sid for sid in include_scan_ids if sid not in exclude_scan_ids]
    return [
        {
            "target_scanner": cfg["target_scanner"],
            "split": cfg["split"],
            "scanner_key": cfg.get("scanner_key"),
            "scan_ids": scan_ids,
            "validation_files": normalize_validation_files(cfg.get("validation_file")),
        }
    ]


def results_dir_for_config(
    cfg: dict[str, Any],
    config_path: Path,
    output_root: Path,
    figure_set: str,
) -> Path:
    subdir = cfg.get("paper_figures_subdir") or cfg.get("results_subdir")
    if subdir:
        return output_root / subdir

    specs = normalize_scanner_specs(cfg)
    if len(specs) == 1:
        spec = specs[0]
        tail = f"{config_path.stem}_{figure_set}"
        return output_root / spec["target_scanner"] / spec["split"] / tail
    return output_root / "combined" / f"{config_path.stem}_{figure_set}"


def load_validation_long(
    validation_dir: Path,
    requested_files: list[str] | None,
    target_scanner: str,
) -> pd.DataFrame:
    if not validation_dir.exists() or not any(validation_dir.glob("*.csv")):
        print(f"  [{target_scanner}] no validation directory at {validation_dir}")
        return pd.DataFrame(columns=["transcript_id", "validation_grade"])

    wide = load_validations(validation_dir, prefix="")
    grade_cols = [c for c in wide.columns if c != "transcript_id"]
    if not grade_cols:
        print(f"  [{target_scanner}] no grade columns in validation files")
        return pd.DataFrame(columns=["transcript_id", "validation_grade"])

    if requested_files is None:
        selected = grade_cols
    else:
        target_stems = [Path(f).stem for f in requested_files]
        missing = [s for s in target_stems if s not in grade_cols]
        if missing:
            raise FileNotFoundError(
                f"validation_files entries not found in {validation_dir}: {missing}. "
                f"Available: {grade_cols}"
            )
        selected = target_stems

    if not selected:
        return pd.DataFrame(columns=["transcript_id", "validation_grade"])

    long = (
        wide[["transcript_id"] + selected]
        .melt(
            id_vars="transcript_id",
            var_name="source_file",
            value_name="validation_grade",
        )
    )
    long["validation_grade"] = pd.to_numeric(long["validation_grade"], errors="coerce")
    long = long.dropna(subset=["validation_grade"])

    conflict_ids = (
        long.groupby("transcript_id")["validation_grade"]
        .nunique()
        .pipe(lambda s: s[s > 1].index.tolist())
    )
    if conflict_ids:
        print(
            f"  [{target_scanner}] warning: {len(conflict_ids)} conflicting "
            "transcript_id(s) across validation files; kept first"
        )

    out = (
        long.drop_duplicates("transcript_id", keep="first")[
            ["transcript_id", "validation_grade"]
        ]
        .reset_index(drop=True)
    )
    per_file = long.groupby("source_file")["transcript_id"].nunique().to_dict()
    print(
        f"  [{target_scanner}] validation: {len(out):,} unique grades "
        f"across {len(selected)} file(s) ({per_file})"
    )
    return out


def load_block(spec: dict[str, Any]) -> pd.DataFrame:
    target_scanner = spec["target_scanner"]
    split = spec["split"]
    requested_scan_ids = list(spec.get("scan_ids") or [])
    requested_scanner_key = spec.get("scanner_key")
    requested_validation = normalize_validation_files(spec.get("validation_files"))

    base = PROJECT_ROOT / "evals" / "scans" / target_scanner / split
    scan_dir = base / "scan-results"
    validation_dir = base / "validation"

    print(f"\n[{target_scanner}/{split}] loading {scan_dir.relative_to(PROJECT_ROOT)}")
    raw = load_scan_results_for_figures(scan_dir).copy()
    raw["target_scanner"] = target_scanner
    raw["split"] = split
    raw["scan_id"] = raw["scanner_source"].str.removeprefix("scan_id=")
    raw["eval_file"] = raw["transcript_source_uri"].fillna("").map(
        lambda p: Path(p).name if p else None
    )
    raw["scanner_model"] = raw["scanner_model"].map(shorten_model)
    raw["eval_generation_model"] = raw["transcript_model"].map(shorten_model)
    raw["benchmark"] = raw["transcript_task_set"].map(
        lambda b: BENCHMARK_ALIASES.get(b, b)
    )
    raw["scanner_label"] = (
        raw["scanner_model"].fillna("(no model)").astype(str)
        + " - "
        + raw["scan_id"].str[:6]
    )

    present_keys = sorted(raw["scanner_key"].dropna().unique().tolist())
    if requested_scanner_key is not None:
        resolved_key = requested_scanner_key
    elif len(present_keys) == 1:
        resolved_key = present_keys[0]
    else:
        raise RuntimeError(
            f"[{target_scanner}] multiple scanner_keys present ({present_keys}); "
            "set `scanner_key` in config."
        )
    raw = raw[raw["scanner_key"] == resolved_key].copy()

    if requested_scan_ids:
        before = raw["scan_id"].nunique()
        raw = raw[raw["scan_id"].isin(requested_scan_ids)].copy()
        missing = set(requested_scan_ids) - set(raw["scan_id"].unique())
        if missing:
            print(f"  [{target_scanner}] warning: missing scan_ids {sorted(missing)}")
        print(f"  [{target_scanner}] scan_ids: {raw['scan_id'].nunique()} of {before} kept")
    else:
        print(f"  [{target_scanner}] scan_ids: all {raw['scan_id'].nunique()} included")

    validation = load_validation_long(validation_dir, requested_validation, target_scanner)
    if validation.empty:
        raw["validation_grade"] = np.nan
    else:
        raw = raw.merge(validation, on="transcript_id", how="left")

    print(
        f"  [{target_scanner}] rows: {len(raw):,}; "
        f"with validation: {int(raw['validation_grade'].notna().sum()):,}"
    )
    return raw


def load_combined(specs: list[dict[str, Any]]) -> pd.DataFrame:
    frames = [load_block(spec) for spec in specs]
    keep_cols = [
        "target_scanner",
        "split",
        "scanner_key",
        "scan_id",
        "scanner_label",
        "scanner_model",
        "scan_timestamp",
        "transcript_id",
        "value",
        "value_num",
        "benchmark",
        "eval_generation_model",
        "eval_file",
        "transcript_score",
        "transcript_success",
        "validation_grade",
    ]
    combined = pd.concat(
        [df.reindex(columns=keep_cols) for df in frames],
        ignore_index=True,
    )
    print(
        f"\nCombined frame: {len(combined):,} rows; "
        f"{combined['target_scanner'].nunique()} target scanner(s); "
        f"{combined['scan_id'].nunique()} scan_id(s); "
        f"{combined['transcript_id'].nunique():,} transcript(s)"
    )
    return combined


def ordered_split_benchmark_keys(df: pd.DataFrame) -> list[tuple[str, str]]:
    split_order = ["dev", "test"]
    available_splits = [s for s in split_order if s in set(df["split"].dropna())]
    available_splits += [
        s for s in df["split"].dropna().unique().tolist() if s not in split_order
    ]
    keys: list[tuple[str, str]] = []
    for split in available_splits:
        benches = (
            df[df["split"] == split]
            .dropna(subset=["benchmark"])["benchmark"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        keys.extend((split, bench) for bench in benches)
    return keys


def grade_counts(series: pd.Series) -> dict[str, int]:
    nums = pd.to_numeric(series, errors="coerce").dropna()
    counts = nums.astype(int).value_counts().sort_index()
    return {f"n_grade_{int(g)}": int(c) for g, c in counts.items()}


def save_overview(
    combined: pd.DataFrame,
    threshold: int,
    save_table: Callable[..., None],
) -> None:
    group_cols = [
        "target_scanner",
        "scan_id",
        "scanner_label",
        "scanner_model",
        "benchmark",
        "eval_generation_model",
        "eval_file",
    ]
    rows = []
    for key, group in combined.groupby(group_cols, dropna=False):
        validated = group.dropna(subset=["validation_grade"])
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "n_scanned": int(len(group)),
                "n_validated": int(len(validated)),
                **grade_counts(group["value_num"]),
                "scanner_violation_rate": violation_rate(group["value_num"], threshold),
                "validation_violation_rate": (
                    violation_rate(validated["validation_grade"], threshold)
                    if not validated.empty
                    else float("nan")
                ),
            }
        )
    overview = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    grade_cols = sorted(
        [c for c in overview.columns if c.startswith("n_grade_")],
        key=lambda c: int(c.removeprefix("n_grade_")),
    )
    if grade_cols:
        overview[grade_cols] = overview[grade_cols].fillna(0).astype(int)
        overview = overview[[c for c in overview.columns if c not in grade_cols] + grade_cols]
    save_table(overview, "overview")


def scale_color(color: str, factor: float) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))


def model_shade(model_name: str) -> float:
    s = str(model_name).lower()
    return 0.65 if "gpt-5" in s or "gpt5" in s or "5.4" in s else 1.0


def plot_grouped_stacked_bars(
    ax,
    ordered_keys: list[tuple[str, str]],
    scanner_models: list[str],
    panel_df: pd.DataFrame,
    all_grades: list[int],
    title: str,
) -> None:
    n_models = max(len(scanner_models), 1)
    n_keys = len(ordered_keys)
    x = np.arange(n_keys)
    group_width = 0.9
    bar_width = group_width / n_models
    offsets = [
        (-group_width / 2) + bar_width / 2 + i * bar_width
        for i in range(n_models)
    ]
    xtick_labels = [f"{split}\n{bench}" for split, bench in ordered_keys]

    group_n = []
    for split, bench in ordered_keys:
        per_bar_ns = [
            panel_df[
                (panel_df["split"] == split)
                & (panel_df["benchmark"] == bench)
                & (panel_df["scanner_model"] == model)
            ]["value_num"]
            .dropna()
            .shape[0]
            for model in scanner_models
        ]
        group_n.append(max(per_bar_ns) if per_bar_ns else 0)

    for m_idx, model in enumerate(scanner_models):
        positions = x + offsets[m_idx]
        props_by_grade = {g: [] for g in all_grades}
        for split, bench in ordered_keys:
            sub = panel_df[
                (panel_df["split"] == split)
                & (panel_df["benchmark"] == bench)
                & (panel_df["scanner_model"] == model)
            ]["value_num"].dropna().astype(int)
            total = len(sub)
            for grade in all_grades:
                props_by_grade[grade].append((sub == grade).sum() / total if total else 0)

        bottom = np.zeros(n_keys)
        for grade in reversed(all_grades):
            heights = np.array(props_by_grade[grade])
            color = scale_color(SCORE_COLORS.get(grade, "#999999"), model_shade(model))
            ax.bar(
                positions,
                heights,
                bar_width,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.4,
            )
            bottom += heights

    for xi, n in zip(x, group_n):
        if n > 0:
            ax.text(xi, 1.005, f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-0.5, max(n_keys - 0.5, 0.5))
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=10)


def plot_grade_distribution(
    combined: pd.DataFrame,
    save_fig: Callable[..., None],
) -> None:
    target_scanners = ordered_values(combined["target_scanner"], PREFERRED_SCANNER_ORDER)
    if not target_scanners:
        print("No scans loaded; skipping grade_distribution.")
        return

    ordered_keys = ordered_split_benchmark_keys(combined)
    all_grades = [
        g for g in sorted(combined["value_num"].dropna().astype(int).unique()) if g >= 1
    ]
    if not all_grades:
        print("No non-zero scanner grades; skipping grade_distribution.")
        return

    all_models = ordered_values(combined["scanner_model"], PREFERRED_MODEL_ORDER)
    fig, axes = plt.subplots(
        len(target_scanners),
        1,
        figsize=(max(4.0, 0.5 * len(ordered_keys) + 2), 2.8 * len(target_scanners)),
        sharey=True,
        squeeze=False,
    )

    for idx, target in enumerate(target_scanners):
        ax = axes[idx][0]
        panel_df = combined[combined["target_scanner"] == target]
        scanner_models = [m for m in all_models if m in set(panel_df["scanner_model"].dropna())]
        panel_keys = [
            (s, b)
            for (s, b) in ordered_keys
            if not panel_df[
                (panel_df["split"] == s) & (panel_df["benchmark"] == b)
            ]["value_num"].dropna().empty
        ]
        plot_grouped_stacked_bars(ax, panel_keys, scanner_models, panel_df, all_grades, target)
        ax.set_ylabel("Proportion")

    grade_handles = [
        plt.Rectangle((0, 0), 1, 1, color=SCORE_COLORS.get(g, "#999999"))
        for g in all_grades
    ]
    ref_color = SCORE_COLORS.get(all_grades[len(all_grades) // 2], "#999999")
    model_handles = [
        plt.Rectangle((0, 0), 1, 1, color=scale_color(ref_color, model_shade(m)))
        for m in all_models
    ]
    grade_legend = fig.legend(
        grade_handles,
        [str(g) for g in all_grades],
        title="Grade",
        loc="lower center",
        bbox_to_anchor=(0.3, -0.07),
        ncol=len(all_grades),
        frameon=False,
    )
    fig.add_artist(grade_legend)
    fig.legend(
        model_handles,
        all_models,
        title="Scanner Model (shade)",
        loc="lower center",
        bbox_to_anchor=(0.75, -0.07),
        ncol=max(len(all_models), 1),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, "grade_distribution")
    plt.close(fig)


def plot_human_grade_distribution(
    combined: pd.DataFrame,
    save_fig: Callable[..., None],
) -> None:
    human = (
        combined.dropna(subset=["validation_grade"])[
            ["target_scanner", "split", "transcript_id", "benchmark", "validation_grade"]
        ]
        .drop_duplicates(["target_scanner", "split", "transcript_id"])
        .copy()
    )
    if human.empty:
        print("No human-validated transcripts; skipping grade_distribution_human.")
        return
    human["validation_int"] = human["validation_grade"].astype(int)

    target_scanners = ordered_values(human["target_scanner"], PREFERRED_SCANNER_ORDER)
    ordered_keys = ordered_split_benchmark_keys(human)
    all_grades = [g for g in sorted(human["validation_int"].unique()) if g >= 1]
    if not all_grades:
        print("No non-zero human grades; skipping grade_distribution_human.")
        return

    fig, axes = plt.subplots(
        len(target_scanners),
        1,
        figsize=(max(4.0, 0.5 * len(ordered_keys) + 2), 2.8 * len(target_scanners)),
        sharey=True,
        squeeze=False,
    )
    for idx, target in enumerate(target_scanners):
        ax = axes[idx][0]
        panel_df = human[human["target_scanner"] == target]
        panel_keys = [
            (s, b)
            for (s, b) in ordered_keys
            if not panel_df[(panel_df["split"] == s) & (panel_df["benchmark"] == b)].empty
        ]
        x = np.arange(len(panel_keys))
        props_by_grade = {g: [] for g in all_grades}
        group_n = []
        for split, bench in panel_keys:
            cell = panel_df[
                (panel_df["split"] == split) & (panel_df["benchmark"] == bench)
            ]["validation_int"]
            total = len(cell)
            group_n.append(total)
            for grade in all_grades:
                props_by_grade[grade].append((cell == grade).sum() / total if total else 0)

        bottom = np.zeros(len(panel_keys))
        for grade in reversed(all_grades):
            heights = np.array(props_by_grade[grade])
            ax.bar(
                x,
                heights,
                0.8,
                bottom=bottom,
                color=SCORE_COLORS.get(grade, "#999999"),
                edgecolor="white",
                linewidth=0.4,
            )
            bottom += heights
        for xi, n in zip(x, group_n):
            if n:
                ax.text(xi, 1.005, f"n={n}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n{b}" for s, b in panel_keys], rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.5, max(len(panel_keys) - 0.5, 0.5))
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Proportion")
        ax.set_title(target, fontsize=10)

    grade_handles = [
        plt.Rectangle((0, 0), 1, 1, color=SCORE_COLORS.get(g, "#999999"))
        for g in all_grades
    ]
    fig.legend(
        grade_handles,
        [str(g) for g in all_grades],
        title="Grade",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(all_grades),
        frameon=False,
    )
    fig.suptitle("Human-labeled grade distribution by benchmark", y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, "grade_distribution_human")
    plt.close(fig)


def metric_row(
    target: np.ndarray,
    prediction: np.ndarray,
    threshold: int,
    weighting: str,
    population_prediction: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    cm = confusion_matrix(target, prediction, GRADE_LEVELS)
    if weighting == "weighted":
        if population_prediction is None:
            population_prediction = prediction
        wt_p, wt_n = stratified_ipw_weights(
            population_prediction,
            prediction,
            threshold=threshold,
        )
        cm_for_kappa = weighted_confusion_matrix(
            cm,
            threshold=threshold,
            wt_p=wt_p,
            wt_n=wt_n,
        )
        metrics = weighted_threshold_metrics(
            target,
            prediction,
            threshold=threshold,
            wt_p=wt_p,
            wt_n=wt_n,
        )
        ci = bootstrap_weighted_metrics(
            target,
            prediction,
            scanner_population=population_prediction,
            threshold=threshold,
        )
        kappa = quadratic_weighted_kappa(cm_for_kappa)
    else:
        metrics = threshold_metrics(target, prediction, threshold=threshold)
        ci = bootstrap_threshold_metrics(target, prediction, threshold=threshold)
        kappa = quadratic_weighted_kappa(cm)

    row = {**metrics, "quadratic_weighted_kappa": kappa}
    for key, (lo, hi) in ci.items():
        row[f"{key}_lo"] = lo
        row[f"{key}_hi"] = hi
    return row, cm


def format_metrics_table(df: pd.DataFrame, weighting: str) -> pd.DataFrame:
    out = format_metric_ci_columns(df, CI_METRIC_COLS)
    if weighting == "weighted":
        out = format_numeric_columns(out, ["wt_p", "wt_n"], decimals=2)
    return out


def pooled_confusion_and_metrics(
    combined: pd.DataFrame,
    threshold: int,
    weighting: str,
    save_fig: Callable[..., None],
    save_table: Callable[..., None],
) -> None:
    valid = combined.dropna(subset=["value_num", "validation_grade"]).copy()
    if valid.empty:
        print("No rows with both scanner and validation grades; skipping pooled metrics.")
        return
    valid["scanner_int"] = valid["value_num"].astype(int)
    valid["validation_int"] = valid["validation_grade"].astype(int)
    population = combined.dropna(subset=["value_num"]).copy()
    population["scanner_int"] = population["value_num"].astype(int)

    pooled_keys = (
        valid.sort_values(["target_scanner", "split", "scanner_model", "scan_timestamp"])
        .drop_duplicates(["target_scanner", "split", "scanner_model"])[
            ["target_scanner", "split", "scanner_model"]
        ]
        .to_dict("records")
    )
    n_panels = len(pooled_keys)
    n_cols = min(4 if weighting == "weighted" else 3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 5.0 * n_rows),
        squeeze=False,
    )
    rows = []
    for idx, info in enumerate(pooled_keys):
        ax = axes[idx // n_cols][idx % n_cols]
        mask = (
            (valid["target_scanner"] == info["target_scanner"])
            & (valid["split"] == info["split"])
            & (valid["scanner_model"] == info["scanner_model"])
        )
        cell = valid[mask]
        pop_cell = population[
            (population["target_scanner"] == info["target_scanner"])
            & (population["split"] == info["split"])
            & (population["scanner_model"] == info["scanner_model"])
        ]
        target = cell["validation_int"].to_numpy()
        pred = cell["scanner_int"].to_numpy()
        row_metrics, cm = metric_row(
            target,
            pred,
            threshold,
            weighting,
            pop_cell["scanner_int"].to_numpy(),
        )
        kappa = row_metrics["quadratic_weighted_kappa"]
        f1 = row_metrics["f1"]
        title_bits = [
            f"{info['target_scanner']} / {info['split']}",
            f"{info['scanner_model']}",
            f"n={row_metrics['n']}",
        ]
        if weighting == "weighted":
            title_bits[-1] += (
                f" (wt_p={row_metrics['wt_p']:.2f}, wt_n={row_metrics['wt_n']:.2f})"
                if pd.notna(row_metrics["wt_p"]) and pd.notna(row_metrics["wt_n"])
                else " (wt_p=-, wt_n=-)"
            )
        title_bits.append(
            f"qwk={kappa:.3f}, F1={f1:.2f}"
            if pd.notna(kappa) and pd.notna(f1)
            else "qwk=-, F1=-"
        )
        draw_cm(
            ax,
            cm,
            xlabel="Scanner",
            ylabel="Validation",
            title="\n".join(title_bits),
        )
        rows.append(
            {
                "target_scanner": info["target_scanner"],
                "split": info["split"],
                "scanner_model": info["scanner_model"],
                "n_scans": int(cell["scan_id"].nunique()),
                **row_metrics,
            }
        )

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle(f"Confusion matrices vs validation grade ({weighting} metrics)", y=1.0)
    fig.tight_layout()
    save_fig(fig, "confusion_matrices_pooled")
    plt.close(fig)

    pooled_df = pd.DataFrame(rows)
    save_table(pooled_df, "pooled_metrics")
    save_table(format_metrics_table(pooled_df, weighting), "pooled_metrics_formatted")


def benchmark_metrics(
    combined: pd.DataFrame,
    threshold: int,
    weighting: str,
    save_table: Callable[..., None],
) -> None:
    scanner_graded = combined.dropna(subset=["value_num"]).copy()
    if scanner_graded.empty:
        print("No scanner-graded rows; skipping benchmark_metrics.")
        return

    keys = (
        scanner_graded.sort_values(
            ["target_scanner", "split", "benchmark", "scanner_model", "scan_timestamp"]
        )
        .drop_duplicates(["target_scanner", "split", "scanner_model", "benchmark"])[
            ["target_scanner", "split", "scanner_model", "benchmark"]
        ]
        .to_dict("records")
    )
    rows = []
    for info in keys:
        cell = scanner_graded[
            (scanner_graded["target_scanner"] == info["target_scanner"])
            & (scanner_graded["split"] == info["split"])
            & (scanner_graded["scanner_model"] == info["scanner_model"])
            & (scanner_graded["benchmark"] == info["benchmark"])
        ]
        validated = cell.dropna(subset=["validation_grade"])
        counts = {
            "n_samples": int(len(cell)),
            "n_flagged": int((cell["value_num"] >= threshold).sum()),
            "n_validated": int(len(validated)),
            "n_h_flagged": int((validated["validation_grade"] >= threshold).sum()),
        }
        if validated.empty:
            base_metrics = {
                "n": 0,
                "accuracy": float("nan"),
                "sensitivity": float("nan"),
                "specificity": float("nan"),
                "precision": float("nan"),
                "f1": float("nan"),
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "quadratic_weighted_kappa": float("nan"),
            }
            for col in CI_METRIC_COLS:
                base_metrics[f"{col}_lo"] = float("nan")
                base_metrics[f"{col}_hi"] = float("nan")
            if weighting == "weighted":
                base_metrics.update(
                    {
                        "n_weighted": 0.0,
                        "wt_p": float("nan"),
                        "wt_n": float("nan"),
                        "tp_w": 0.0,
                        "tn_w": 0.0,
                        "fp_w": 0.0,
                        "fn_w": 0.0,
                    }
                )
        else:
            base_metrics, _ = metric_row(
                validated["validation_grade"].astype(int).to_numpy(),
                validated["value_num"].astype(int).to_numpy(),
                threshold,
                weighting,
                cell["value_num"].astype(int).to_numpy(),
            )
        rows.append({**info, **counts, **base_metrics})

    df = pd.DataFrame(rows)
    save_table(df, "benchmark_metrics")
    save_table(format_metrics_table(df, weighting), "benchmark_metrics_formatted")


def scanner_agreement(
    combined: pd.DataFrame,
    save_fig: Callable[..., None],
    save_table: Callable[..., None],
) -> None:
    per_model = (
        combined.dropna(subset=["value_num"])
        .groupby(
            ["target_scanner", "split", "transcript_id", "scanner_model"],
            dropna=False,
        )["value_num"]
        .mean()
        .reset_index()
    )
    block_keys = (
        per_model[["target_scanner", "split"]]
        .drop_duplicates()
        .sort_values(["target_scanner", "split"])
        .to_dict("records")
    )
    panels = []
    for block in block_keys:
        target, split = block["target_scanner"], block["split"]
        block_pm = per_model[
            (per_model["target_scanner"] == target) & (per_model["split"] == split)
        ]
        model_order = sorted(block_pm["scanner_model"].dropna().unique().tolist())
        if len(model_order) < 2:
            print(f"Only {len(model_order)} scanner_model loaded for {target}/{split}; skipping agreement.")
            continue
        pivot = block_pm.pivot_table(
            index="transcript_id",
            columns="scanner_model",
            values="value_num",
            aggfunc="first",
        )
        for model_a, model_b in itertools.combinations(model_order, 2):
            panels.append(
                {
                    "target_scanner": target,
                    "split": split,
                    "model_a": model_a,
                    "model_b": model_b,
                    "joint": pivot[[model_a, model_b]].dropna(),
                }
            )

    if not panels:
        return

    n_panels = len(panels)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.0 * n_rows), squeeze=False)
    rows = []
    for idx, panel in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        joint = panel["joint"]
        if joint.empty:
            ax.axis("off")
            continue
        ga = np.floor(joint[panel["model_a"]]).astype(int).to_numpy()
        gb = np.floor(joint[panel["model_b"]]).astype(int).to_numpy()
        cm = confusion_matrix(ga, gb, GRADE_LEVELS)
        kappa = quadratic_weighted_kappa(cm)
        kappa_lo, kappa_hi = bootstrap_kappa(ga, gb)
        n = int(cm.sum())
        agree = int(np.trace(cm))
        draw_cm(
            ax,
            cm,
            xlabel=panel["model_b"],
            ylabel=panel["model_a"],
            title=(
                f"{panel['target_scanner']} / {panel['split']}\n"
                f"{panel['model_a']} vs {panel['model_b']}\n"
                f"n={n}, agree={agree / n:.1%}, qwk={kappa:.3f}"
            ),
        )
        rows.append(
            {
                "target_scanner": panel["target_scanner"],
                "split": panel["split"],
                "model_a": panel["model_a"],
                "model_b": panel["model_b"],
                "n_paired": n,
                "agree_rate": agree / n if n else float("nan"),
                "mean_abs_diff": float(np.abs(ga - gb).mean()) if n else float("nan"),
                "quadratic_weighted_kappa": kappa,
                "quadratic_weighted_kappa_lo": kappa_lo,
                "quadratic_weighted_kappa_hi": kappa_hi,
            }
        )
    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.tight_layout()
    save_fig(fig, "scanner_agreement_all_blocks")
    plt.close(fig)

    pair_df = pd.DataFrame(rows).sort_values(
        ["target_scanner", "split", "model_a", "model_b"]
    )
    save_table(pair_df, "scanner_agreement_all_blocks")
    display_pair = format_metric_ci_columns(pair_df, ["quadratic_weighted_kappa"])
    display_pair = format_numeric_columns(display_pair, ["agree_rate"], decimals=1, as_percent=True)
    display_pair = format_numeric_columns(display_pair, ["mean_abs_diff"], decimals=2)
    save_table(display_pair, "scanner_agreement_all_blocks_formatted")


def composite_analysis(
    combined: pd.DataFrame,
    threshold: int,
    weighting: str,
    save_fig: Callable[..., None],
    save_table: Callable[..., None],
) -> None:
    per_model = (
        combined.dropna(subset=["value_num"])
        .groupby(
            ["target_scanner", "split", "transcript_id", "scanner_model"],
            dropna=False,
        )["value_num"]
        .mean()
        .reset_index()
    )
    val_per_block = (
        combined.dropna(subset=["validation_grade"])[
            ["target_scanner", "split", "transcript_id", "validation_grade"]
        ]
        .drop_duplicates(["target_scanner", "split", "transcript_id"])
    )
    block_keys = (
        per_model[["target_scanner", "split"]]
        .drop_duplicates()
        .sort_values(["target_scanner", "split"])
        .to_dict("records")
    )

    all_metrics_rows = []
    all_bucket_rows = []
    composite_blocks = []

    for block in block_keys:
        target, split = block["target_scanner"], block["split"]
        block_pm = per_model[
            (per_model["target_scanner"] == target) & (per_model["split"] == split)
        ]
        model_order = sorted(block_pm["scanner_model"].dropna().unique().tolist())
        if len(model_order) < 2:
            print(f"Only {len(model_order)} scanner_model loaded for {target}/{split}; skipping composites.")
            continue

        pivot = block_pm.pivot_table(
            index="transcript_id",
            columns="scanner_model",
            values="value_num",
            aggfunc="first",
        )
        grade_count = pivot.notna().sum(axis=1)
        composite = pivot[grade_count >= 2].copy()
        if composite.empty:
            continue
        composite["n_scanner_models"] = grade_count[grade_count >= 2]
        composite["composite_floor_mean"] = np.floor(
            composite[model_order].mean(axis=1, skipna=True)
        ).astype(int)
        composite["composite_ceil_mean"] = np.ceil(
            composite[model_order].mean(axis=1, skipna=True)
        ).astype(int)
        composite["composite_max"] = composite[model_order].max(axis=1, skipna=True).astype(int)
        composite_blocks.append(
            composite.reset_index()
            .assign(target_scanner=target, split=split)[
                [
                    "target_scanner",
                    "split",
                    "transcript_id",
                    "composite_floor_mean",
                    "composite_ceil_mean",
                    "composite_max",
                ]
            ]
        )

        block_val = val_per_block[
            (val_per_block["target_scanner"] == target)
            & (val_per_block["split"] == split)
        ][["transcript_id", "validation_grade"]]
        has_validation = not block_val.empty
        composite_eval = (
            composite.merge(block_val.set_index("transcript_id"), left_index=True, right_index=True, how="left")
            if has_validation
            else composite
        )

        if has_validation:
            scored = composite_eval.dropna(subset=["validation_grade"]).copy()
            if not scored.empty:
                scored["validation_int"] = scored["validation_grade"].astype(int)
                fig, axes = plt.subplots(1, len(COMPOSITES), figsize=(4.6 * len(COMPOSITES), 4.4), squeeze=False)
                metric_rows = []
                for ax, (name, col) in zip(axes[0], COMPOSITES):
                    target_arr = scored["validation_int"].to_numpy()
                    pred = scored[col].astype(int).to_numpy()
                    pop_pred = composite_eval[col].astype(int).to_numpy()
                    row_metrics, cm = metric_row(target_arr, pred, threshold, weighting, pop_pred)
                    draw_cm(
                        ax,
                        cm,
                        xlabel="Composite",
                        ylabel="Validation",
                        title=(
                            f"composite = {name}\n"
                            f"n={row_metrics['n']}\n"
                            f"qwk={row_metrics['quadratic_weighted_kappa']:.3f}"
                        ),
                    )
                    metric_rows.append(
                        {
                            "target_scanner": target,
                            "split": split,
                            "composite": name,
                            **row_metrics,
                        }
                    )
                fig.suptitle(
                    f"{target} / {split} - Composite vs validation ({weighting} metrics)",
                    y=0.98,
                )
                fig.tight_layout()
                save_fig(fig, f"composite_vs_validation__{target}__{split}")
                plt.close(fig)
                all_metrics_rows.extend(metric_rows)

        grade_only = composite[model_order]
        flags = grade_only >= threshold
        flags_neg = grade_only < threshold
        n_pos = flags.sum(axis=1)
        n_neg = flags_neg.sum(axis=1)
        n_total = composite["n_scanner_models"]
        bucket = pd.Series("any-disagree", index=composite.index)
        bucket[(n_pos == n_total) | (n_neg == n_total)] = "all-agree"
        composite["agreement_bucket"] = bucket

        summary_rows = []
        for bucket_name, group in composite.groupby("agreement_bucket"):
            base = {
                "target_scanner": target,
                "split": split,
                "bucket": bucket_name,
                "n_transcripts": int(len(group)),
            }
            if not has_validation:
                summary_rows.append({**base, "composite": "-"})
                continue
            with_val = group.merge(
                block_val.set_index("transcript_id"),
                left_index=True,
                right_index=True,
                how="left",
            ).dropna(subset=["validation_grade"])
            if with_val.empty:
                summary_rows.append(
                    {
                        **base,
                        "composite": "-",
                        "n_validated": 0,
                        "validation_violation_rate": float("nan"),
                    }
                )
                continue
            target_arr = with_val["validation_grade"].astype(int).to_numpy()
            common = {
                **base,
                "n_validated": int(len(with_val)),
                "validation_violation_rate": float((target_arr >= threshold).mean()),
            }

            def emit(name: str, col: str) -> dict[str, Any]:
                pred = with_val[col].astype(int).to_numpy()
                pop_pred = group[col].astype(int).to_numpy()
                row_metrics, _ = metric_row(target_arr, pred, threshold, weighting, pop_pred)
                return {**common, "composite": name, **row_metrics}

            if bucket_name == "all-agree":
                summary_rows.append(emit("-", COMPOSITES[0][1]))
            else:
                for name, col in COMPOSITES:
                    summary_rows.append(emit(name, col))

        if summary_rows:
            bucket_df = pd.DataFrame(summary_rows).sort_values(["bucket", "composite"])
            all_bucket_rows.extend(summary_rows)
            save_table(bucket_df, f"agreement_partition__{target}__{split}")
            display_bucket = format_metrics_table(bucket_df.drop(columns=["target_scanner", "split"]), weighting)
            display_bucket = format_numeric_columns(
                display_bucket,
                ["validation_violation_rate"],
                decimals=1,
                as_percent=True,
            )
            save_table(display_bucket, f"agreement_partition__{target}__{split}_formatted")

    if all_metrics_rows:
        metrics_df = pd.DataFrame(all_metrics_rows)
        save_table(metrics_df, "composite_metrics_all_blocks")
        save_table(format_metrics_table(metrics_df, weighting), "composite_metrics_all_blocks_formatted")
    if all_bucket_rows:
        bucket_df = pd.DataFrame(all_bucket_rows)
        save_table(bucket_df, "agreement_partition_all_blocks")
        display_bucket = format_metrics_table(bucket_df, weighting)
        display_bucket = format_numeric_columns(
            display_bucket,
            ["validation_violation_rate"],
            decimals=1,
            as_percent=True,
        )
        save_table(display_bucket, "agreement_partition_all_blocks_formatted")

    composite_long = (
        pd.concat(composite_blocks, ignore_index=True)
        if composite_blocks
        else pd.DataFrame()
    )
    plot_composite_grade_distributions(combined, composite_long, save_fig)


def plot_composite_grade_distributions(
    combined: pd.DataFrame,
    composite_long: pd.DataFrame,
    save_fig: Callable[..., None],
) -> None:
    if composite_long.empty:
        return
    transcript_meta = (
        combined.dropna(subset=["benchmark"])[
            ["target_scanner", "split", "transcript_id", "benchmark"]
        ]
        .drop_duplicates(["target_scanner", "split", "transcript_id"])
    )
    plot_df = composite_long.merge(
        transcript_meta,
        on=["target_scanner", "split", "transcript_id"],
        how="left",
    )
    target_scanners = ordered_values(plot_df["target_scanner"], PREFERRED_SCANNER_ORDER)
    ordered_keys = ordered_split_benchmark_keys(plot_df)
    grades_present = sorted(
        pd.concat([plot_df[c] for _, c in COMPOSITES]).dropna().astype(int).unique()
    )
    plot_grades = [g for g in grades_present if g >= 1]
    if not plot_grades:
        return

    for name, col in COMPOSITES:
        fig, axes = plt.subplots(
            len(target_scanners),
            1,
            figsize=(max(4.0, 0.5 * len(ordered_keys) + 2), 2.8 * len(target_scanners)),
            sharey=True,
            squeeze=False,
        )
        for idx, target in enumerate(target_scanners):
            ax = axes[idx][0]
            panel_df = plot_df[plot_df["target_scanner"] == target]
            panel_keys = [
                (s, b)
                for (s, b) in ordered_keys
                if not panel_df[
                    (panel_df["split"] == s) & (panel_df["benchmark"] == b)
                ][col].dropna().empty
            ]
            x = np.arange(len(panel_keys))
            props_by_grade = {g: [] for g in plot_grades}
            group_n = []
            for split, bench in panel_keys:
                cell = panel_df[
                    (panel_df["split"] == split) & (panel_df["benchmark"] == bench)
                ][col].dropna().astype(int)
                total = len(cell)
                group_n.append(total)
                for grade in plot_grades:
                    props_by_grade[grade].append((cell == grade).sum() / total if total else 0)
            bottom = np.zeros(len(panel_keys))
            for grade in reversed(plot_grades):
                heights = np.array(props_by_grade[grade])
                ax.bar(
                    x,
                    heights,
                    0.8,
                    bottom=bottom,
                    color=SCORE_COLORS.get(grade, "#999999"),
                    edgecolor="white",
                    linewidth=0.4,
                )
                bottom += heights
            for xi, n in zip(x, group_n):
                if n:
                    ax.text(xi, 1.005, f"n={n}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{s}\n{b}" for s, b in panel_keys], rotation=45, ha="right", fontsize=8)
            ax.set_xlim(-0.5, max(len(panel_keys) - 0.5, 0.5))
            ax.set_ylim(0, 1.12)
            ax.set_ylabel("Proportion")
            ax.set_title(target, fontsize=10)
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=SCORE_COLORS.get(g, "#999999"))
            for g in plot_grades
        ]
        fig.legend(
            handles,
            [str(g) for g in plot_grades],
            title="Grade",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=len(plot_grades),
            frameon=False,
        )
        fig.suptitle(f"Composite grade distribution by benchmark - {name}", y=0.995)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        save_fig(fig, f"composite_grade_distribution__{col}")
        plt.close(fig)


def run_combined_figures(
    cfg: dict[str, Any],
    config_path: Path,
    output_root: Path,
    weighting: str,
) -> None:
    specs = normalize_scanner_specs(cfg)
    results_dir = results_dir_for_config(cfg, config_path, output_root, "combined")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_fig, save_table = make_savers(results_dir)
    threshold = int(cfg.get("violation_threshold", 2))

    print(f"CONFIG      = {config_path.relative_to(PROJECT_ROOT)}")
    print(f"THRESHOLD   = {threshold}")
    print(f"WEIGHTING   = {weighting}")
    print(f"RESULTS_DIR = {path_label(results_dir)}")

    combined = load_combined(specs)
    save_overview(combined, threshold, save_table)
    plot_grade_distribution(combined, save_fig)
    plot_human_grade_distribution(combined, save_fig)
    pooled_confusion_and_metrics(combined, threshold, weighting, save_fig, save_table)
    benchmark_metrics(combined, threshold, weighting, save_table)
    scanner_agreement(combined, save_fig, save_table)
    composite_analysis(combined, threshold, weighting, save_fig, save_table)


def wrate(indicator: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    indicator = np.asarray(indicator, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    return float((indicator * weights).sum() / total) if total > 0 else float("nan")


def stratified_bootstrap(
    val_df: pd.DataFrame,
    strata_cols: list[str],
    stat_fn: Callable[[pd.DataFrame], dict[str, float]],
    *,
    n_boot: int,
    ci: float,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    d = val_df.reset_index(drop=True)
    groups = [g.index.to_numpy() for _, g in d.groupby(strata_cols, dropna=False)]
    point = stat_fn(d)
    samples = {k: np.full(n_boot, np.nan) for k in point}
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        sel = np.concatenate([rng.choice(idx, size=idx.size, replace=True) for idx in groups])
        boot = stat_fn(d.iloc[sel])
        for k, v in boot.items():
            samples[k][b] = v
    alpha = (1.0 - ci) / 2.0
    out = {}
    for k, p in point.items():
        arr = samples[k][~np.isnan(samples[k])]
        if arr.size and not np.isnan(p):
            lo, hi = float(np.quantile(arr, alpha)), float(np.quantile(arr, 1 - alpha))
            out[k] = (p, min(p, lo), max(p, hi))
        else:
            out[k] = (p, float("nan"), float("nan"))
    return out


def build_empirical_criterion(
    criterion: str,
    spec: dict[str, Any],
    threshold: int,
    stratifier_model: str,
    secondary_model: str,
) -> pd.DataFrame:
    base = PROJECT_ROOT / "evals" / "scans" / criterion / spec["split"]
    raw = load_scan_results_for_figures(base / "scan-results").copy()
    raw["scan_id"] = raw["scanner_source"].str.removeprefix("scan_id=")
    raw["scanner_model"] = raw["scanner_model"].map(shorten_model)
    raw["benchmark"] = raw["transcript_task_set"].map(lambda b: BENCHMARK_ALIASES.get(b, b))

    keys = sorted(raw["scanner_key"].dropna().unique())
    resolved = spec.get("scanner_key") or (keys[0] if len(keys) == 1 else None)
    if resolved is None:
        raise RuntimeError(f"[{criterion}] multiple scanner_keys {keys}; set scanner_key.")
    raw = raw[raw["scanner_key"] == resolved].copy()

    validation_files = normalize_validation_files(spec.get("validation_files"))
    if not validation_files:
        print(f"  [{criterion}] no validation files; skipping empirical criterion")
        return pd.DataFrame()

    val_stems = [Path(f).stem for f in validation_files]
    gpt = raw[raw["scanner_model"] == stratifier_model]
    gpt_val_ids = sorted(
        {
            sid
            for sid in gpt["scan_id"].unique()
            if any(str(sid) in stem for stem in val_stems)
        }
    )
    if not gpt_val_ids:
        # Fall back to explicit scan_ids from config when validation filenames
        # do not embed scan IDs.
        gpt_val_ids = [
            sid
            for sid in list(spec.get("scan_ids") or [])
            if sid in set(gpt["scan_id"].unique())
        ]
    if not gpt_val_ids:
        print(f"  [{criterion}] no {stratifier_model} scan_id matched validation files")
        return pd.DataFrame()

    pop = (
        gpt[gpt["scan_id"].isin(gpt_val_ids)]
        .groupby(["transcript_id", "benchmark"], as_index=False)["value_num"]
        .max()
        .rename(columns={"value_num": "gpt_grade"})
    )
    secondary = (
        raw[raw["scanner_model"] == secondary_model]
        .groupby("transcript_id", as_index=False)["value_num"]
        .max()
        .rename(columns={"value_num": "sonnet_grade"})
    )
    pop = pop.merge(secondary, on="transcript_id", how="left")

    validation = load_validation_long(base / "validation", validation_files, criterion)
    validation = validation.rename(columns={"validation_grade": "human_grade"})
    pop = pop.merge(validation, on="transcript_id", how="left")
    pop.insert(0, "criterion", criterion)
    print(
        f"  [{criterion}] universe={len(pop):,}; "
        f"validated={int(pop['human_grade'].notna().sum()):,}; "
        f"stratifier scans={gpt_val_ids}"
    )
    return pop


def run_empirical_paper_figures(
    cfg: dict[str, Any],
    config_path: Path,
    output_root: Path,
) -> None:
    specs = {s["target_scanner"]: dict(s) for s in normalize_scanner_specs(cfg)}
    criteria = sorted(specs)
    if not criteria:
        raise ValueError("Empirical paper figures require at least one scanner spec.")

    paper_cfg = cfg.get("paper_figures") or {}
    threshold = int(cfg.get("violation_threshold", 2))
    exclude_benchmarks = set(paper_cfg.get("exclude_benchmarks", ["gpqa_diamond", "hle"]))
    stratifier_model = paper_cfg.get("stratifier_model", "gpt-5.4")
    secondary_model = paper_cfg.get("secondary_model", "sonnet-4.6")
    composites = paper_cfg.get("composites", ["max", "floor_mean"])
    boot_n = int(paper_cfg.get("boot_n", 2000))
    boot_ci = float(paper_cfg.get("boot_ci", 0.95))
    boot_seed = int(paper_cfg.get("boot_seed", 0))

    results_dir = output_root / "combined" / f"{config_path.stem}_empirical"
    if cfg.get("paper_figures_subdir"):
        results_dir = output_root / cfg["paper_figures_subdir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    save_fig, save_table = make_savers(results_dir)

    print(f"\nEMPIRICAL PAPER FIGURES")
    print(f"RESULTS_DIR      = {path_label(results_dir)}")
    print(f"STRATIFIER_MODEL = {stratifier_model}")
    print(f"SECONDARY_MODEL  = {secondary_model}")
    print(f"COMPOSITES       = {composites}")

    frames = [
        build_empirical_criterion(
            criterion,
            specs[criterion],
            threshold,
            stratifier_model,
            secondary_model,
        )
        for criterion in criteria
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        print("No empirical frames built; skipping empirical paper figures.")
        return

    tx = pd.concat(frames, ignore_index=True)
    tx = tx[~tx["benchmark"].isin(exclude_benchmarks)].copy()
    if tx.empty:
        print("All empirical rows excluded by benchmark filter.")
        return
    if tx["sonnet_grade"].isna().any():
        missing = int(tx["sonnet_grade"].isna().sum())
        print(f"Warning: {missing} empirical rows lack {secondary_model}; dropping them.")
        tx = tx.dropna(subset=["sonnet_grade"]).copy()

    tx["gpt_flag"] = (tx["gpt_grade"] >= threshold).astype(int)
    tx["sonnet_flag"] = (tx["sonnet_grade"] >= threshold).astype(float)
    tx["human_viol"] = (tx["human_grade"] >= threshold).astype(float)
    tx["validated"] = tx["human_grade"].notna()
    tx["max_grade"] = tx[["gpt_grade", "sonnet_grade"]].max(axis=1)
    tx["floor_mean_grade"] = np.floor(tx[["gpt_grade", "sonnet_grade"]].mean(axis=1))
    for comp in composites:
        tx[f"{comp}_flag"] = (tx[f"{comp}_grade"] >= threshold).astype(int)

    pop_counts = tx.groupby(["criterion", "benchmark", "gpt_flag"]).size().rename("N_pop")
    val_counts = (
        tx[tx["validated"]]
        .groupby(["criterion", "benchmark", "gpt_flag"])
        .size()
        .rename("n_val")
    )
    strata = pd.concat([pop_counts, val_counts], axis=1).reset_index()
    strata["n_val"] = strata["n_val"].fillna(0).astype(int)
    strata["weight"] = np.where(strata["n_val"] > 0, strata["N_pop"] / strata["n_val"], np.nan)
    strata["samp_frac"] = strata["n_val"] / strata["N_pop"]
    tx = tx.merge(
        strata[["criterion", "benchmark", "gpt_flag", "weight", "N_pop", "n_val"]],
        on=["criterion", "benchmark", "gpt_flag"],
        how="left",
    )
    save_table(strata, "empirical_sampling_strata")

    criterion_colors = {
        c: color
        for c, color in zip(
            criteria,
            ["#4477AA", "#AA3377", "#66CCEE", "#CCBB44", "#222255", "#999933"],
        )
    }

    for comp in composites:
        figure_empirical_violation_rates(
            tx,
            comp,
            criteria,
            criterion_colors,
            boot_n,
            boot_ci,
            boot_seed,
            save_fig,
            save_table,
        )
    figure_empirical_severity(
        tx,
        criteria,
        threshold,
        stratifier_model,
        secondary_model,
        boot_n,
        boot_ci,
        boot_seed,
        save_fig,
        save_table,
    )


def figure_empirical_violation_rates(
    tx: pd.DataFrame,
    comp: str,
    criteria: list[str],
    criterion_colors: dict[str, str],
    boot_n: int,
    boot_ci: float,
    boot_seed: int,
    save_fig: Callable[..., None],
    save_table: Callable[..., None],
) -> None:
    flag_col = f"{comp}_flag"
    label = "max across scanners" if comp == "max" else "floor(mean) across scanners"
    rows = []
    for (criterion, benchmark), group in tx.groupby(["criterion", "benchmark"]):
        scan = stratified_bootstrap(
            group,
            ["benchmark"],
            lambda d: {"rate": float(d[flag_col].mean())},
            n_boot=boot_n,
            ci=boot_ci,
            seed=boot_seed,
        )
        s_rate, s_lo, s_hi = scan["rate"]
        validated = group[group["validated"]]
        present = set(group["gpt_flag"].unique())
        covered = bool(present) and all((validated["gpt_flag"] == s).sum() > 0 for s in present)
        if not validated.empty and covered:
            hh = stratified_bootstrap(
                validated,
                ["gpt_flag"],
                lambda d: {"rate": wrate(d["human_viol"], d["weight"])},
                n_boot=boot_n,
                ci=boot_ci,
                seed=boot_seed,
            )
            h_rate, h_lo, h_hi = hh["rate"]
            naive = float(validated["human_viol"].mean())
        else:
            h_rate = h_lo = h_hi = naive = float("nan")
        rows.append(
            {
                "criterion": criterion,
                "benchmark": benchmark,
                "n_pop": int(len(group)),
                "n_val": int(len(validated)),
                "scanner_rate": s_rate,
                "scanner_lo": s_lo,
                "scanner_hi": s_hi,
                "human_rate_ipw": h_rate,
                "human_lo": h_lo,
                "human_hi": h_hi,
                "human_rate_naive": naive,
            }
        )

    rates = pd.DataFrame(rows).sort_values(["benchmark", "criterion"]).reset_index(drop=True)
    save_table(rates, f"violation_rates_empirical_{comp}")
    benchmarks = sorted(rates["benchmark"].unique())
    offsets = np.linspace(-0.27, 0.27, len(criteria))
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for offset, criterion in zip(offsets, criteria):
        sm = rates[rates["criterion"] == criterion].set_index("benchmark").reindex(benchmarks)
        xpos = np.arange(len(benchmarks)) + offset
        color = criterion_colors[criterion]
        ax.errorbar(
            xpos,
            sm["scanner_rate"],
            yerr=[
                np.clip(sm["scanner_rate"] - sm["scanner_lo"], 0, None),
                np.clip(sm["scanner_hi"] - sm["scanner_rate"], 0, None),
            ],
            fmt="o",
            color=color,
            ms=6,
            lw=1.8,
            capsize=3,
            label=display_label(criterion),
        )
        ax.scatter(xpos, sm["human_rate_ipw"], marker="x", color=color, s=36, zorder=3, alpha=0.85)

    ax.set_xticks(np.arange(len(benchmarks)))
    ax.set_xticklabels([display_label(b) for b in benchmarks], rotation=20, ha="right")
    for i in range(len(benchmarks)):
        if i % 2:
            ax.axvspan(i - 0.5, i + 0.5, color="#f2f2f2", zorder=0)
    ax.set_xlim(-0.5, len(benchmarks) - 0.5)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel("Violation rate")
    ax.set_title("Scanner Flagged and Human-confirmed Violation Rate by Benchmark and Criterion")
    legend_handles = [
        Line2D([], [], color=criterion_colors[c], marker="o", lw=1.8, label=display_label(c))
        for c in criteria
    ] + [
        Line2D([], [], color="#555555", marker="o", ls="", label=f"composite flag rate ({label})"),
        Line2D([], [], color="#555555", marker="x", ls="", label="human-confirmed (IPW-adj.)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save_fig(fig, f"violation_rates_by_benchmark_{comp}")
    plt.close(fig)


def figure_empirical_severity(
    tx: pd.DataFrame,
    criteria: list[str],
    threshold: int,
    stratifier_model: str,
    secondary_model: str,
    boot_n: int,
    boot_ci: float,
    boot_seed: int,
    save_fig: Callable[..., None],
    save_table: Callable[..., None],
) -> None:
    model_color = {secondary_model: "#BF5700", stratifier_model: "#1B5E20"}
    display_models = [stratifier_model, secondary_model]
    model_cols = {stratifier_model: "gpt_flag", secondary_model: "sonnet_flag"}
    human_grades = sorted(int(g) for g in tx.loc[tx["validated"], "human_grade"].unique())

    rows = []
    for criterion in criteria:
        validated = tx[(tx["criterion"] == criterion) & tx["validated"]].copy()
        if validated.empty:
            continue
        validated["hg_int"] = validated["human_grade"].astype(int)
        for model in display_models:
            col = model_cols[model]

            def stat(d: pd.DataFrame, col: str = col) -> dict[str, float]:
                out = {}
                for grade in human_grades:
                    sub = d[d["hg_int"] == grade]
                    out[f"g{grade}"] = wrate(sub[col], sub["weight"]) if len(sub) else np.nan
                return out

            res = stratified_bootstrap(
                validated,
                ["benchmark", "gpt_flag"],
                stat,
                n_boot=boot_n,
                ci=boot_ci,
                seed=boot_seed,
            )
            for grade in human_grades:
                n = int((validated["hg_int"] == grade).sum())
                if n == 0:
                    continue
                rate, lo, hi = res[f"g{grade}"]
                rows.append(
                    {
                        "criterion": criterion,
                        "scanner_model": model,
                        "human_grade": grade,
                        "n": n,
                        "flag_rate": rate,
                        "lo": lo,
                        "hi": hi,
                    }
                )

    severity = pd.DataFrame(rows)
    if severity.empty:
        return
    save_table(severity, "flag_rate_by_severity_empirical")

    n_panels = len(criteria)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.5, 3.8 * n_rows),
        sharey=True,
        squeeze=False,
    )
    for ax, criterion in zip(axes.ravel(), criteria):
        for offset, model in zip((-0.08, 0.08), display_models):
            sm = severity[
                (severity["criterion"] == criterion)
                & (severity["scanner_model"] == model)
            ].sort_values("human_grade")
            if sm.empty:
                continue
            color = model_color[model]
            x = sm["human_grade"] + offset
            ax.errorbar(
                x,
                sm["flag_rate"],
                yerr=[sm["flag_rate"] - sm["lo"], sm["hi"] - sm["flag_rate"]],
                fmt="none",
                ecolor=color,
                elinewidth=1.1,
                alpha=0.6,
            )
            ax.plot(x, sm["flag_rate"], "-", color=color, lw=1.2, alpha=0.7)
            ax.scatter(x, sm["flag_rate"], s=18 + 2.2 * sm["n"], color=color, zorder=3, label=model)
        ax.axvline(threshold - 0.5, color="#999999", lw=1, ls="--")
        ax.set_title(display_label(criterion))
        ax.set_xticks(human_grades)
        ax.set_ylim(-0.04, 1.04)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e8e8e8", lw=0.6)
        ax.set_axisbelow(True)

    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    axes[0][0].legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("Human Grade")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(scanner flag), 95% CI")
    fig.suptitle("Scanner flag rate by human-graded severity")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, "flag_rate_by_severity")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAML config file. Existing analysis/configs/*.yaml files are supported.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "results" / "paper_figures",
        help="Root directory for generated paper figures.",
    )
    parser.add_argument(
        "--weighting",
        choices=["weighted", "unweighted"],
        default=None,
        help="Metric weighting. Defaults to config metric_weighting or weighted.",
    )
    parser.add_argument(
        "--figure-set",
        choices=["combined", "empirical", "all"],
        default="all",
        help=(
            "combined: notebook diagnostic figures; empirical: final empirical "
            "paper figures from scanner_paper_figures.ipynb; all: both. "
            "Default: all."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    weighting = args.weighting or cfg.get("metric_weighting", "weighted")
    if weighting not in {"weighted", "unweighted"}:
        raise ValueError("metric weighting must be `weighted` or `unweighted`")

    output_root = args.output_root.resolve()
    if args.figure_set in {"combined", "all"}:
        run_combined_figures(cfg, config_path, output_root, weighting)
    if args.figure_set in {"empirical", "all"}:
        run_empirical_paper_figures(cfg, config_path, output_root)


if __name__ == "__main__":
    main()
