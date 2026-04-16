"""
CHD-VLM Results Analysis
========================
Statistical analysis of CHD-CXR VLM evaluation results produced by
the evaluation pipeline. Takes a results DataFrame (or a CSV path) and
computes a comprehensive set of classification and calibration metrics,
bootstrap confidence intervals, and pairwise significance tests.

Usage
-----
    from src.analysis.analyze import run_full_analysis, print_summary
    import pandas as pd

    df = pd.read_csv("results/combined_all_models.csv")
    analysis = run_full_analysis(df)

    print(analysis["metrics"])           # one row per model × prompt
    print(analysis["per_class_metrics"]) # one row per (condition, class)
    print(analysis["mcnemar_results"])   # pairwise significance tests

The returned ``analysis`` dict is consumed directly by ``visualization/visualize.py``.
"""

from __future__ import annotations

import logging
import warnings
from itertools import combinations
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

LABELS: list[str] = ["ASD", "VSD", "PDA", "Normal"]


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] = LABELS,
) -> dict:
    """Compute a full suite of classification metrics for one condition.

    Parameters
    ----------
    y_true:
        Array of ground-truth label strings.
    y_pred:
        Array of predicted label strings. ``None`` values are treated as
        incorrect predictions (replaced with a dummy label).
    labels:
        Ordered list of class names.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``macro_f1``, ``kappa``, ``roc_auc``,
        ``per_class_f1``, ``per_class_sensitivity``, ``per_class_specificity``,
        ``confusion_matrix``, ``n_total``, ``n_parsed``, ``parse_rate``.
    """
    _NONE_PLACEHOLDER = "__NONE__"
    y_pred = np.where(pd.isnull(y_pred), _NONE_PLACEHOLDER, y_pred).astype(str)

    n_total = len(y_true)
    n_valid = int((y_pred != _NONE_PLACEHOLDER).sum())
    parse_rate = n_valid / n_total if n_total else 0.0

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    kappa = float(cohen_kappa_score(y_true, y_pred))

    per_class_f1 = {
        label: float(f)
        for label, f in zip(
            labels,
            f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        )
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_sensitivity = {}
    per_class_specificity = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per_class_sensitivity[label] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        per_class_specificity[label] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "n_total": n_total,
        "n_parsed": n_valid,
        "parse_rate": round(parse_rate, 4),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "kappa": round(kappa, 4),
        "roc_auc": None,
        "per_class_f1": {k: round(v, 4) for k, v in per_class_f1.items()},
        "per_class_sensitivity": {k: round(v, 4) for k, v in per_class_sensitivity.items()},
        "per_class_specificity": {k: round(v, 4) for k, v in per_class_specificity.items()},
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted arrays.
    metric_fn:
        A function ``(y_true, y_pred) -> float``.
    n_bootstrap:
        Number of bootstrap resamples.
    alpha:
        Significance level; returns ``(1-alpha)*100%`` CI.
    random_state:
        RNG seed for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        ``(point_estimate, ci_lower, ci_upper)``
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            scores.append(np.nan)

    scores = np.array(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return point, np.nan, np.nan

    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return round(point, 4), round(lower, 4), round(upper, 4)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 for use as a bootstrap metric function."""
    y_pred = np.where(pd.isnull(y_pred), "__NONE__", y_pred).astype(str)
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.where(pd.isnull(y_pred), "__NONE__", y_pred).astype(str)
    return accuracy_score(y_true, y_pred)


# ---------------------------------------------------------------------------
# Pairwise McNemar tests + BH FDR correction
# ---------------------------------------------------------------------------


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> tuple[float, float]:
    """McNemar's test comparing two classifiers on the same test set.

    Returns
    -------
    tuple[float, float]
        ``(chi2_statistic, p_value)``
    """
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())

    if b + c == 0:
        return 0.0, 1.0

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - stats.chi2.cdf(chi2, df=1))
    return round(float(chi2), 4), round(p_value, 6)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    order = np.argsort(p_values)
    corrected = np.empty(n)
    cumulative_min = np.inf
    for i in range(n - 1, -1, -1):
        rank = order[i] + 1
        adjusted = p_values[order[i]] * n / rank
        cumulative_min = min(cumulative_min, adjusted)
        corrected[order[i]] = min(cumulative_min, 1.0)
    return corrected


def compute_pairwise_mcnemar(
    results_df: pd.DataFrame,
    group_cols: list[str] = ("model_name", "prompt_id"),
) -> pd.DataFrame:
    """Run pairwise McNemar tests across all (model, prompt) combinations.

    Returns
    -------
    pd.DataFrame
        Columns: ``condition_a``, ``condition_b``, ``chi2``, ``p_raw``,
        ``p_bh``, ``significant_05``, ``significant_01``.
    """
    groups = results_df.groupby(list(group_cols))
    condition_keys = list(groups.groups.keys())

    rows = []
    for key_a, key_b in combinations(condition_keys, 2):
        df_a = groups.get_group(key_a).set_index("sample_id")
        df_b = groups.get_group(key_b).set_index("sample_id")
        shared_ids = df_a.index.intersection(df_b.index)
        if len(shared_ids) == 0:
            continue

        y_true = df_a.loc[shared_ids, "true_label"].values
        pred_a = df_a.loc[shared_ids, "predicted_label"].fillna("__NONE__").values
        pred_b = df_b.loc[shared_ids, "predicted_label"].fillna("__NONE__").values

        chi2, p_val = mcnemar_test(y_true, pred_a, pred_b)
        rows.append(
            {
                "condition_a": " / ".join(str(k) for k in key_a),
                "condition_b": " / ".join(str(k) for k in key_b),
                "n_shared": len(shared_ids),
                "chi2": chi2,
                "p_raw": p_val,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["condition_a", "condition_b", "n_shared", "chi2",
                     "p_raw", "p_bh", "significant_05", "significant_01"]
        )

    mcnemar_df = pd.DataFrame(rows)
    mcnemar_df["p_bh"] = benjamini_hochberg(mcnemar_df["p_raw"].values)
    mcnemar_df["significant_05"] = mcnemar_df["p_bh"] < 0.05
    mcnemar_df["significant_01"] = mcnemar_df["p_bh"] < 0.01
    return mcnemar_df.sort_values("p_bh").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Calibration metrics (requires probability columns)
# ---------------------------------------------------------------------------


def compute_calibration_metrics(
    results_df: pd.DataFrame,
    prob_cols: Optional[dict[str, str]] = None,
    n_bins: int = 10,
) -> Optional[dict]:
    """Compute calibration metrics (ECE, MCE, Brier score) and reliability data.

    Returns ``None`` if probability columns are absent.
    """
    if prob_cols is None:
        prob_cols = {label: f"prob_{label}" for label in LABELS}

    missing = [col for col in prob_cols.values() if col not in results_df.columns]
    if missing:
        return None

    df = results_df.dropna(subset=list(prob_cols.values()) + ["true_label"])
    if df.empty:
        return None

    prob_matrix = df[list(prob_cols.values())].values
    label_to_idx = {label: i for i, label in enumerate(prob_cols.keys())}
    y_idx = df["true_label"].map(label_to_idx).values
    n, k = prob_matrix.shape

    try:
        roc_auc = float(
            roc_auc_score(
                np.eye(k)[y_idx],
                prob_matrix,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        roc_auc = None

    one_hot = np.eye(k)[y_idx]
    brier = float(np.mean((prob_matrix - one_hot) ** 2))

    max_probs = prob_matrix.max(axis=1)
    predicted_idx = prob_matrix.argmax(axis=1)
    correct = (predicted_idx == y_idx).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_sum = 0.0
    mce = 0.0
    reliability = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (max_probs >= lo) & (max_probs < hi)
        if hi == 1.0:
            mask = max_probs >= lo
        count = int(mask.sum())
        if count == 0:
            reliability.append(((lo + hi) / 2, 0.0, 0))
            continue
        avg_conf = float(max_probs[mask].mean())
        frac_correct = float(correct[mask].mean())
        gap = abs(avg_conf - frac_correct)
        ece_sum += gap * count / n
        mce = max(mce, gap)
        reliability.append(((lo + hi) / 2, frac_correct, count))

    return {
        "ece": round(ece_sum, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "reliability": reliability,
        "n_calibration_samples": n,
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_full_analysis(
    results_df: Union[pd.DataFrame, str],
    group_cols: list[str] = ("model_name", "prompt_id"),
    n_bootstrap: int = 2000,
    bootstrap_metrics: Optional[list[str]] = None,
) -> dict:
    """Run all analyses and return a structured results dict.

    Parameters
    ----------
    results_df:
        Either a pandas DataFrame (output of the evaluation pipeline) or a
        path to a CSV file. Multiple experiment CSVs can be pre-concatenated.
    group_cols:
        Columns that together define one experimental condition
        (default: model × prompt).
    n_bootstrap:
        Number of bootstrap resamples for confidence intervals.
    bootstrap_metrics:
        Which metrics to bootstrap. Defaults to ``["macro_f1", "accuracy"]``.

    Returns
    -------
    dict
        ``{
            "metrics":            pd.DataFrame,   # one row per condition
            "per_class_metrics":  pd.DataFrame,   # one row per (condition, class)
            "bootstrap_cis":      pd.DataFrame,   # CI bounds per condition
            "mcnemar_results":    pd.DataFrame,   # pairwise significance tests
            "calibration":        dict | None,    # calibration per condition
            "confusion_matrices": dict,           # (model, prompt) -> np.ndarray
        }``
    """
    if bootstrap_metrics is None:
        bootstrap_metrics = ["macro_f1", "accuracy"]

    if isinstance(results_df, str):
        results_df = pd.read_csv(results_df)

    required_cols = {"true_label", "predicted_label"}
    if not required_cols.issubset(results_df.columns):
        raise ValueError(
            f"results_df must contain columns {required_cols}. "
            f"Got: {list(results_df.columns)}"
        )

    if "sample_id" not in results_df.columns:
        results_df = results_df.copy()
        results_df["sample_id"] = results_df.index.astype(str)

    metrics_rows = []
    per_class_rows = []
    ci_rows = []
    confusion_matrices = {}
    calibration_results = {}

    group_cols_list = list(group_cols)
    for keys, group in results_df.groupby(group_cols_list, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        condition_dict = dict(zip(group_cols_list, keys))

        y_true = group["true_label"].values
        y_pred = group["predicted_label"].values

        m = compute_classification_metrics(y_true, y_pred)
        row = {**condition_dict, **{
            k: v for k, v in m.items()
            if k not in ("per_class_f1", "per_class_sensitivity",
                         "per_class_specificity", "confusion_matrix")
        }}
        metrics_rows.append(row)
        confusion_matrices[keys] = m["confusion_matrix"]

        for label in LABELS:
            per_class_rows.append({
                **condition_dict,
                "class": label,
                "f1": m["per_class_f1"].get(label, 0.0),
                "sensitivity": m["per_class_sensitivity"].get(label, 0.0),
                "specificity": m["per_class_specificity"].get(label, 0.0),
            })

        ci_row = dict(condition_dict)
        if "macro_f1" in bootstrap_metrics:
            est, lo, hi = bootstrap_ci(y_true, y_pred, _macro_f1, n_bootstrap)
            ci_row.update(
                macro_f1_point=est, macro_f1_ci_lower=lo, macro_f1_ci_upper=hi
            )
        if "accuracy" in bootstrap_metrics:
            est, lo, hi = bootstrap_ci(y_true, y_pred, _accuracy, n_bootstrap)
            ci_row.update(
                accuracy_point=est, accuracy_ci_lower=lo, accuracy_ci_upper=hi
            )
        ci_rows.append(ci_row)

        cal = compute_calibration_metrics(group)
        if cal is not None:
            calibration_results[keys] = cal
            metrics_rows[-1]["roc_auc"] = cal.get("roc_auc")

    metrics_df = pd.DataFrame(metrics_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    ci_df = pd.DataFrame(ci_rows)

    logger.info(
        "Analysis complete: %d experimental conditions, %d total samples.",
        len(metrics_df),
        len(results_df),
    )

    if len(metrics_rows) >= 2:
        mcnemar_df = compute_pairwise_mcnemar(results_df, group_cols=group_cols_list)
    else:
        mcnemar_df = pd.DataFrame(
            columns=["condition_a", "condition_b", "n_shared", "chi2",
                     "p_raw", "p_bh", "significant_05", "significant_01"]
        )

    return {
        "metrics": metrics_df,
        "per_class_metrics": per_class_df,
        "bootstrap_cis": ci_df,
        "mcnemar_results": mcnemar_df,
        "calibration": calibration_results if calibration_results else None,
        "confusion_matrices": confusion_matrices,
    }


# ---------------------------------------------------------------------------
# Cost summary helper
# ---------------------------------------------------------------------------


def cost_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute API cost summary from a results DataFrame.

    Convenience wrapper over :class:`src.evaluation.cost_tracker.CostTracker`.
    Returns an empty DataFrame if no API cost columns are present.

    Returns
    -------
    pd.DataFrame
        Columns: model_name, prompt_id, n_calls, total_input_tokens,
        total_output_tokens, total_cost_usd, avg_cost_per_sample_usd.
    """
    from src.evaluation.cost_tracker import CostTracker
    tracker = CostTracker.from_results_df(results_df)
    return tracker.summary()


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def print_summary(analysis: dict) -> None:
    """Print a human-readable summary of the analysis results."""
    metrics = analysis["metrics"]
    display_cols = [
        c for c in
        ["model_name", "prompt_id", "n_total", "parse_rate", "accuracy",
         "macro_f1", "kappa", "roc_auc"]
        if c in metrics.columns
    ]
    print("\n===  Classification Metrics  ===")
    print(metrics[display_cols].to_string(index=False))

    ci_df = analysis["bootstrap_cis"]
    if not ci_df.empty and "macro_f1_ci_lower" in ci_df.columns:
        print("\n===  Bootstrap 95% CI (Macro-F1)  ===")
        ci_cols = [c for c in ["model_name", "prompt_id",
                                "macro_f1_point", "macro_f1_ci_lower",
                                "macro_f1_ci_upper"] if c in ci_df.columns]
        print(ci_df[ci_cols].to_string(index=False))

    mcnemar = analysis["mcnemar_results"]
    if not mcnemar.empty:
        sig = mcnemar[mcnemar["significant_05"]]
        print(f"\n===  Significant Pairwise Differences (BH p<0.05): {len(sig)}  ===")
        if not sig.empty:
            print(sig[["condition_a", "condition_b", "p_bh"]].to_string(index=False))

    if analysis["calibration"]:
        print("\n===  Calibration (ECE / MCE / Brier)  ===")
        for key, cal in analysis["calibration"].items():
            print(
                f"  {' / '.join(str(k) for k in key)}: "
                f"ECE={cal['ece']:.4f}  MCE={cal['mce']:.4f}  "
                f"Brier={cal['brier_score']:.4f}"
            )
