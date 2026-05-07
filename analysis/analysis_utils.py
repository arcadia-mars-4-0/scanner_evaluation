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
