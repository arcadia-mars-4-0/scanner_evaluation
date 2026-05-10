"""Shared display invariants, metric helpers, and plot helpers for the
scanner-analysis notebooks (`scanner_validation`, `scanner_synthetic`,
`scanner_combined_results`)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# --- Display invariants ---

GRADE_LEVELS: list[int] = [0, 1, 2, 3]

SCORE_COLORS: dict[int, str] = {
    0: "#4393c3",
    1: "#f4d35e",
    2: "#d1495b",
    3: "#7f0000",
}

BENCHMARK_ALIASES: dict[str, str] = {
    "swe_bench_verified_mini": "swe_bench",
}

MODEL_NAME_ALIASES: dict[str, str] = {
    "anthropic/claude-sonnet-4-6": "sonnet-4.6",
    "anthropic/claude-opus-4-6": "opus-4.6",
    "openai/gpt-5.4": "gpt-5.4",
    "openai/gpt-5.4-2026-03-05": "gpt-5.4",
    "openai/gpt-5-mini-2025-08-07": "gpt-5-mini",
    "openai/gpt-5-mini": "gpt-5-mini",
}


def shorten_model(name):
    if name is None:
        return name
    return MODEL_NAME_ALIASES.get(name, name)


# --- Metric helpers ---

def violation_rate(series: pd.Series, threshold: float) -> float:
    """Fraction of numeric values in ``series`` that meet ``>= threshold``."""
    nums = pd.to_numeric(series, errors="coerce").dropna()
    if nums.empty:
        return float("nan")
    return float(nums.ge(threshold).mean())


def confusion_matrix(
    rows: np.ndarray,
    cols: np.ndarray,
    levels: list[int] = GRADE_LEVELS,
) -> np.ndarray:
    idx = {g: i for i, g in enumerate(levels)}
    k = len(levels)
    cm = np.zeros((k, k), dtype=int)
    for r, c in zip(rows, cols):
        if r in idx and c in idx:
            cm[idx[r], idx[c]] += 1
    return cm


def quadratic_weighted_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    k = cm.shape[0]
    if k < 2:
        return float("nan")
    weights = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    observed = cm / n
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    expected = np.outer(row_marg, col_marg)
    denom = (weights * expected).sum()
    if denom == 0:
        return float("nan")
    return 1.0 - (weights * observed).sum() / denom


def threshold_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    threshold: float,
) -> dict:
    tgt = target >= threshold
    pred = prediction >= threshold
    tp = int((pred & tgt).sum())
    tn = int((~pred & ~tgt).sum())
    fp = int((pred & ~tgt).sum())
    fn = int((~pred & tgt).sum())
    n = len(tgt)
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def stratified_ipw_weights(
    scanner_population: np.ndarray,
    scanner_validated: np.ndarray,
    *,
    threshold: float,
) -> tuple[float, float]:
    """Inverse sampling-probability weights for a scanner-stratified validation subset.

    Validation rows were drawn separately from the scanner-positive
    (``scanner >= threshold``) and scanner-negative strata of the full
    scanner-graded population. Each validated row therefore stands in for
    ``stratum_population_size / stratum_validated_size`` population rows.

    Returns ``(wt_p, wt_n)``; either is ``nan`` if its stratum has no
    validated rows (so weighted metrics involving that stratum are undefined).
    """
    pop = np.asarray(scanner_population, dtype=float)
    val = np.asarray(scanner_validated, dtype=float)
    n_pos_pop = int((pop >= threshold).sum())
    n_neg_pop = int((pop < threshold).sum())
    n_pos_val = int((val >= threshold).sum())
    n_neg_val = int((val < threshold).sum())
    wt_p = n_pos_pop / n_pos_val if n_pos_val > 0 else float("nan")
    wt_n = n_neg_pop / n_neg_val if n_neg_val > 0 else float("nan")
    return wt_p, wt_n


def _wcount(count: int, weight: float) -> float:
    """``count * weight``, but treat ``0 * nan`` as ``0`` (empty strata don't poison sums)."""
    if count == 0:
        return 0.0
    return count * weight


def weighted_threshold_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    threshold: float,
    wt_p: float,
    wt_n: float,
) -> dict:
    """Threshold metrics with stratified IPW applied via ``wt_p`` / ``wt_n``.

    Stratification is on the *scanner* (prediction) label: validated rows with
    ``prediction >= threshold`` carry weight ``wt_p``; the rest carry ``wt_n``.
    Sensitivity, specificity, precision, F1, and accuracy are computed from
    the weighted (TP, FP, TN, FN). Raw counts and weights are echoed back.
    """
    tgt = np.asarray(target) >= threshold
    pred = np.asarray(prediction) >= threshold
    tp = int((pred & tgt).sum())
    tn = int((~pred & ~tgt).sum())
    fp = int((pred & ~tgt).sum())
    fn = int((~pred & tgt).sum())
    n = len(tgt)

    tp_w = _wcount(tp, wt_p)
    fp_w = _wcount(fp, wt_p)
    tn_w = _wcount(tn, wt_n)
    fn_w = _wcount(fn, wt_n)
    n_w = tp_w + fp_w + tn_w + fn_w

    def _safe(num: float, den: float) -> float:
        if np.isnan(num) or np.isnan(den) or den == 0:
            return float("nan")
        return num / den

    return {
        "n": n,
        "n_weighted": n_w,
        "wt_p": wt_p,
        "wt_n": wt_n,
        "accuracy": _safe(tp_w + tn_w, n_w),
        "sensitivity": _safe(tp_w, tp_w + fn_w),
        "specificity": _safe(tn_w, tn_w + fp_w),
        "precision": _safe(tp_w, tp_w + fp_w),
        "f1": _safe(2 * tp_w, 2 * tp_w + fp_w + fn_w),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tp_w": tp_w, "tn_w": tn_w, "fp_w": fp_w, "fn_w": fn_w,
    }


# --- Bootstrap CIs ---

BOOT_N: int = 1000
BOOT_SEED: int = 0
BOOT_CI: float = 0.95


def _percentile_ci(samples: np.ndarray, ci: float) -> tuple[float, float]:
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def bootstrap_weighted_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    scanner_population: np.ndarray,
    threshold: float,
    levels: list[int] = GRADE_LEVELS,
    n_boot: int = BOOT_N,
    ci: float = BOOT_CI,
    seed: int | None = BOOT_SEED,
) -> dict[str, tuple[float, float]]:
    """Stratified percentile-bootstrap CIs for the metrics in
    ``weighted_threshold_metrics`` plus ``quadratic_weighted_kappa``.

    Resamples validated rows with replacement *within* each scanner-stratum
    (``prediction >= threshold`` vs ``prediction < threshold``), matching the
    IPW design. Stratum sizes are preserved by construction, so ``wt_p`` /
    ``wt_n`` equal the point-estimate weights on every iteration.
    """
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    pop = np.asarray(scanner_population)

    metric_keys = [
        "accuracy", "sensitivity", "specificity", "precision", "f1",
        "quadratic_weighted_kappa",
    ]
    nan_result = {k: (float("nan"), float("nan")) for k in metric_keys}

    idx_pos = np.flatnonzero(prediction >= threshold)
    idx_neg = np.flatnonzero(prediction < threshold)
    if idx_pos.size == 0 or idx_neg.size == 0:
        return nan_result

    n_pos_pop = int((pop >= threshold).sum())
    n_neg_pop = int((pop < threshold).sum())
    wt_p = n_pos_pop / idx_pos.size
    wt_n = n_neg_pop / idx_neg.size

    rng = np.random.default_rng(seed)
    samples = {k: np.full(n_boot, np.nan) for k in metric_keys}
    for b in range(n_boot):
        sel = np.concatenate([
            rng.choice(idx_pos, size=idx_pos.size, replace=True),
            rng.choice(idx_neg, size=idx_neg.size, replace=True),
        ])
        t_b = target[sel]
        p_b = prediction[sel]
        m = weighted_threshold_metrics(
            t_b, p_b, threshold=threshold, wt_p=wt_p, wt_n=wt_n,
        )
        for k in ("accuracy", "sensitivity", "specificity", "precision", "f1"):
            samples[k][b] = m[k]
        cm = confusion_matrix(t_b, p_b, levels)
        cm_w = weighted_confusion_matrix(
            cm, threshold=threshold, wt_p=wt_p, wt_n=wt_n, levels=levels,
        )
        samples["quadratic_weighted_kappa"][b] = quadratic_weighted_kappa(cm_w)

    return {k: _percentile_ci(samples[k], ci) for k in metric_keys}


def bootstrap_threshold_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    threshold: float,
    levels: list[int] = GRADE_LEVELS,
    n_boot: int = BOOT_N,
    ci: float = BOOT_CI,
    seed: int | None = BOOT_SEED,
) -> dict[str, tuple[float, float]]:
    """Unstratified percentile-bootstrap CIs for the metrics in
    ``threshold_metrics`` plus ``quadratic_weighted_kappa``.

    Resamples paired ``(target, prediction)`` rows uniformly with replacement.
    Use this when no sampling-design correction is needed; for stratified
    designs use ``bootstrap_weighted_metrics`` instead.
    """
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    n = target.size
    metric_keys = [
        "accuracy", "sensitivity", "specificity", "precision", "f1",
        "quadratic_weighted_kappa",
    ]
    if n == 0:
        return {k: (float("nan"), float("nan")) for k in metric_keys}

    rng = np.random.default_rng(seed)
    samples = {k: np.full(n_boot, np.nan) for k in metric_keys}
    for b in range(n_boot):
        sel = rng.integers(0, n, size=n)
        t_b = target[sel]
        p_b = prediction[sel]
        m = threshold_metrics(t_b, p_b, threshold=threshold)
        for k in ("accuracy", "sensitivity", "specificity", "precision", "f1"):
            samples[k][b] = m[k]
        cm = confusion_matrix(t_b, p_b, levels)
        samples["quadratic_weighted_kappa"][b] = quadratic_weighted_kappa(cm)

    return {k: _percentile_ci(samples[k], ci) for k in metric_keys}


def bootstrap_kappa(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    levels: list[int] = GRADE_LEVELS,
    n_boot: int = BOOT_N,
    ci: float = BOOT_CI,
    seed: int | None = BOOT_SEED,
) -> tuple[float, float]:
    """Unstratified percentile-bootstrap CI for ``quadratic_weighted_kappa``
    over paired ``(rows, cols)`` ratings."""
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    n = rows.size
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.full(n_boot, np.nan)
    for b in range(n_boot):
        sel = rng.integers(0, n, size=n)
        cm = confusion_matrix(rows[sel], cols[sel], levels)
        samples[b] = quadratic_weighted_kappa(cm)
    return _percentile_ci(samples, ci)


def format_metric_ci(
    point: float,
    lo: float,
    hi: float,
    *,
    decimals: int = 2,
    missing: str = "—",
) -> str:
    """Render ``point (lo-hi)`` with ``decimals`` digits; ``missing`` if nan."""
    if point is None or (isinstance(point, float) and np.isnan(point)):
        return missing
    p = f"{point:.{decimals}f}"
    if np.isnan(lo) or np.isnan(hi):
        return p
    return f"{p} ({lo:.{decimals}f}-{hi:.{decimals}f})"


def format_metric_ci_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    decimals: int = 2,
    drop_bounds: bool = True,
) -> pd.DataFrame:
    """Replace each metric column with its ``point (lo-hi)`` string, pulling
    bounds from ``f'{col}_lo'`` / ``f'{col}_hi'`` in ``df``. Missing bound
    columns are skipped (the point column is left untouched)."""
    out = df.copy()
    for col in columns:
        lo_col, hi_col = f"{col}_lo", f"{col}_hi"
        if lo_col not in out.columns or hi_col not in out.columns:
            continue
        out[col] = [
            format_metric_ci(p, lo, hi, decimals=decimals)
            for p, lo, hi in zip(out[col], out[lo_col], out[hi_col])
        ]
        if drop_bounds:
            out = out.drop(columns=[lo_col, hi_col])
    return out


def format_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    decimals: int = 2,
    as_percent: bool = False,
    missing: str = "—",
) -> pd.DataFrame:
    """Format selected numeric columns; ``as_percent`` uses ``{:.Nf}%``-style
    percent rendering (decimals = digits after the point)."""
    spec = f".{decimals}{'%' if as_percent else 'f'}"
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[col] = out[col].map(
            lambda v, _s=spec: format(v, _s) if pd.notna(v) else missing
        )
    return out


def weighted_confusion_matrix(
    cm: np.ndarray,
    *,
    threshold: float,
    wt_p: float,
    wt_n: float,
    levels: list[int] = GRADE_LEVELS,
) -> np.ndarray:
    """Scale each column of ``cm`` by its scanner-stratum weight.

    Column ``j`` (= scanner grade ``levels[j]``) is multiplied by ``wt_p`` if
    ``levels[j] >= threshold`` else ``wt_n``. The returned matrix has float
    entries; ``quadratic_weighted_kappa`` accepts it as-is.
    """
    cm_w = cm.astype(float).copy()
    for j, g in enumerate(levels):
        w = wt_p if g >= threshold else wt_n
        cm_w[:, j] *= w
    return cm_w


# --- Plot helpers ---

def draw_cm(
    ax,
    cm: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    levels: list[int] = GRADE_LEVELS,
) -> None:
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(levels)))
    ax.set_yticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_yticklabels(levels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(len(levels)):
        for j in range(len(levels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > vmax / 2 else "black")
    ax.set_title(title, fontsize=9)


# --- Output helpers ---

def make_savers(
    results_dir: Path | None,
) -> tuple[Callable[..., None], Callable[..., None]]:
    """Return (save_fig, save_table) bound to ``results_dir``.

    When ``results_dir`` is ``None`` both savers become no-ops, matching the
    notebooks' "don't save outputs" mode.
    """
    def save_fig(fig, name: str) -> None:
        if results_dir is None:
            return
        path = results_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  saved figure → {path}")

    def save_table(df: pd.DataFrame, name: str, *, index: bool = False) -> None:
        if results_dir is None:
            return
        path = results_dir / f"{name}.csv"
        df.to_csv(path, index=index)
        print(f"  saved table  → {path}")

    return save_fig, save_table
