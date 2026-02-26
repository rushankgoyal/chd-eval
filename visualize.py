"""
CHD-VLM Results Visualization
==============================
Generates publication-quality figures from the analysis dict produced by
``analyze.run_full_analysis()``.

Individual plot functions each return a ``matplotlib.figure.Figure`` object
(allowing further customisation or embedding in notebooks). The convenience
function :func:`save_all_figures` runs all plots and writes them to disk.

Usage
-----
    from analyze import run_full_analysis
    from visualize import save_all_figures
    import pandas as pd

    df = pd.read_csv("results_all_models.csv")
    analysis = run_full_analysis(df)
    save_all_figures(analysis, output_dir="figures")

Holistic figure
---------------
:func:`plot_holistic_dashboard` produces a single multi-panel figure suitable
for an abstract or poster, combining the most informative views side-by-side.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

LABELS: list[str] = ["ASD", "VSD", "PDA", "Normal"]

# Colour palette: one colour per CHD class, consistent across all figures
CLASS_COLORS: dict[str, str] = {
    "ASD":    "#4C72B0",
    "VSD":    "#DD8452",
    "PDA":    "#55A868",
    "Normal": "#C44E52",
}

# Marker colours per prompt paradigm
PROMPT_COLORS: dict[str, str] = {
    "ZSD": "#1f77b4",
    "RCE": "#ff7f0e",
    "CoT": "#2ca02c",
    "RAC": "#d62728",
}


def _setup_style() -> None:
    """Apply a clean, publication-friendly matplotlib style."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


_setup_style()


def _short_name(full_name: str, max_len: int = 20) -> str:
    """Shorten a HuggingFace model name for display (keep the last segment)."""
    parts = full_name.split("/")
    short = parts[-1] if len(parts) > 1 else full_name
    return short[:max_len] + "…" if len(short) > max_len else short


def _condition_label(row: pd.Series, group_cols: list[str]) -> str:
    """Build a compact display string for a (model, prompt) condition."""
    parts = []
    if "model_name" in group_cols and "model_name" in row:
        parts.append(_short_name(str(row["model_name"])))
    if "prompt_id" in group_cols and "prompt_id" in row:
        parts.append(str(row["prompt_id"]))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 1. Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "",
    labels: list[str] = LABELS,
    normalise: bool = True,
    figsize: tuple = (5, 4),
) -> Figure:
    """Plot a single confusion matrix.

    Parameters
    ----------
    cm:
        Raw confusion matrix (int array, shape K×K), e.g. from
        ``analysis["confusion_matrices"][key]``.
    title:
        Figure title.
    labels:
        Class label names (row / column order matches ``cm``).
    normalise:
        If True, show row-normalised (recall) values; annotate with both
        percentages and raw counts.
    figsize:
        Figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums == 0, 0.0, cm / row_sums.astype(float))
        data = cm_norm
        fmt_fn = lambda v, raw: f"{v:.1%}\n({raw})"
        vmin, vmax = 0.0, 1.0
        cbar_label = "Recall (row-normalised)"
    else:
        data = cm.astype(float)
        fmt_fn = lambda v, raw: str(raw)
        vmin, vmax = 0, cm.max()
        cbar_label = "Count"

    im = ax.imshow(data, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, pad=8)

    # Annotate cells
    thresh = (vmin + vmax) / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if data[i, j] > thresh else "black"
            text = fmt_fn(data[i, j], cm[i, j]) if normalise else str(cm[i, j])
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8, color=color)

    fig.tight_layout()
    return fig


def plot_all_confusion_matrices(
    analysis: dict,
    normalise: bool = True,
    max_cols: int = 4,
) -> Figure:
    """Grid of confusion matrices — one per (model, prompt) condition.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cms = analysis["confusion_matrices"]
    n = len(cms)
    if n == 0:
        raise ValueError("No confusion matrices found in analysis dict.")

    n_cols = min(n, max_cols)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for ax_idx, (key, cm) in enumerate(cms.items()):
        row_i, col_i = divmod(ax_idx, n_cols)
        ax = axes[row_i][col_i]
        title = " / ".join(str(k) for k in key) if isinstance(key, tuple) else str(key)

        if normalise:
            row_sums = cm.sum(axis=1, keepdims=True)
            data = np.where(row_sums == 0, 0.0, cm / row_sums.astype(float))
        else:
            data = cm.astype(float)

        sns.heatmap(
            data, annot=cm, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS,
            vmin=0, vmax=1 if normalise else cm.max(),
            ax=ax, cbar=False, linewidths=0.5, linecolor="lightgrey",
        )
        ax.set_title(_short_name(title, 35), fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

    # Hide unused axes
    for ax_idx in range(n, n_rows * n_cols):
        row_i, col_i = divmod(ax_idx, n_cols)
        axes[row_i][col_i].set_visible(False)

    fig.suptitle("Confusion Matrices (normalised by row)", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Macro-F1 comparison bar chart (prompt sensitivity)
# ---------------------------------------------------------------------------


def plot_macro_f1_bar(
    analysis: dict,
    metric: str = "macro_f1",
    figsize: tuple = (10, 5),
) -> Figure:
    """Grouped bar chart of a metric across models, coloured by prompt.

    One group of bars per model; bars within a group represent prompt variants.
    Bootstrap 95% CI error bars are shown when available.

    Returns
    -------
    matplotlib.figure.Figure
    """
    metrics = analysis["metrics"].copy()
    ci_df = analysis.get("bootstrap_cis", pd.DataFrame())

    if metric not in metrics.columns:
        raise ValueError(f"Metric '{metric}' not found in metrics DataFrame.")

    if "model_name" not in metrics.columns or "prompt_id" not in metrics.columns:
        raise ValueError(
            "plot_macro_f1_bar requires 'model_name' and 'prompt_id' columns."
        )

    # Merge CI bounds if available
    ci_lower_col = f"{metric}_ci_lower"
    ci_upper_col = f"{metric}_ci_upper"
    if not ci_df.empty and ci_lower_col in ci_df.columns:
        merge_on = [c for c in ["model_name", "prompt_id"] if c in ci_df.columns]
        metrics = metrics.merge(ci_df[merge_on + [ci_lower_col, ci_upper_col]],
                                on=merge_on, how="left")

    models = metrics["model_name"].unique()
    prompts = metrics["prompt_id"].unique()
    n_models = len(models)
    n_prompts = len(prompts)

    x = np.arange(n_models)
    bar_width = 0.8 / n_prompts

    fig, ax = plt.subplots(figsize=figsize)

    for p_idx, prompt in enumerate(prompts):
        subset = metrics[metrics["prompt_id"] == prompt].set_index("model_name")
        values = [float(subset.loc[m, metric]) if m in subset.index else 0.0
                  for m in models]
        offsets = x - 0.4 + bar_width * (p_idx + 0.5)

        yerr = None
        if ci_lower_col in metrics.columns:
            lo = [float(subset.loc[m, ci_lower_col]) if m in subset.index else 0.0
                  for m in models]
            hi = [float(subset.loc[m, ci_upper_col]) if m in subset.index else 0.0
                  for m in models]
            yerr = [
                [v - l for v, l in zip(values, lo)],
                [h - v for v, h in zip(values, hi)],
            ]

        color = PROMPT_COLORS.get(str(prompt), None)
        ax.bar(
            offsets, values, width=bar_width, label=str(prompt),
            color=color, yerr=yerr, capsize=3, error_kw={"elinewidth": 1},
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(m) for m in models], rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(
        f"{metric.replace('_', ' ').title()} by Model and Prompt Strategy",
        fontsize=12,
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(title="Prompt", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.axhline(0.70, linestyle="--", color="grey", linewidth=0.8, alpha=0.7,
               label="Clinical utility threshold (F1=0.70)")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Per-class F1 heatmap (models × CHD subtypes)
# ---------------------------------------------------------------------------


def plot_per_class_heatmap(
    analysis: dict,
    metric: str = "f1",
    figsize: tuple = (10, 6),
    prompt_filter: Optional[str] = None,
) -> Figure:
    """Heatmap of per-class metric with conditions on rows and classes on columns.

    Parameters
    ----------
    analysis:
        Output of :func:`analyze.run_full_analysis`.
    metric:
        Per-class metric to display: ``"f1"``, ``"sensitivity"``, or
        ``"specificity"``.
    prompt_filter:
        If provided, only show rows for this prompt_id (useful when you want
        one heatmap per prompt).

    Returns
    -------
    matplotlib.figure.Figure
    """
    per_class = analysis["per_class_metrics"].copy()
    if "class" not in per_class.columns:
        raise ValueError("per_class_metrics must contain a 'class' column.")
    if metric not in per_class.columns:
        raise ValueError(f"Metric '{metric}' not found in per_class_metrics.")

    if prompt_filter and "prompt_id" in per_class.columns:
        per_class = per_class[per_class["prompt_id"] == prompt_filter]

    # Build condition label
    id_cols = [c for c in ["model_name", "prompt_id"] if c in per_class.columns]
    per_class["condition"] = per_class[id_cols].apply(
        lambda r: " / ".join(_short_name(str(v)) for v in r), axis=1
    )

    pivot = per_class.pivot_table(index="condition", columns="class", values=metric)
    pivot = pivot.reindex(columns=[c for c in LABELS if c in pivot.columns])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
        linewidths=0.5, linecolor="lightgrey",
        ax=ax, cbar_kws={"label": metric.title()},
    )
    title_suffix = f" (prompt: {prompt_filter})" if prompt_filter else ""
    ax.set_title(
        f"Per-Class {metric.title()} by Condition{title_suffix}", fontsize=12
    )
    ax.set_xlabel("CHD Class", fontsize=10)
    ax.set_ylabel("Model / Prompt", fontsize=10)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    ax.tick_params(axis="x", labelsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Bootstrap CI lollipop chart
# ---------------------------------------------------------------------------


def plot_bootstrap_ci(
    analysis: dict,
    metric: str = "macro_f1",
    figsize: tuple = (8, 5),
) -> Figure:
    """Lollipop chart of point estimates with 95% bootstrap CI error bars.

    Conditions are sorted by point estimate (descending).

    Returns
    -------
    matplotlib.figure.Figure
    """
    ci_df = analysis.get("bootstrap_cis", pd.DataFrame())
    point_col = f"{metric}_point"
    lo_col = f"{metric}_ci_lower"
    hi_col = f"{metric}_ci_upper"

    if ci_df.empty or point_col not in ci_df.columns:
        # Fall back to the metrics table without CIs
        ci_df = analysis["metrics"].copy()
        if metric not in ci_df.columns:
            raise ValueError(f"Metric '{metric}' not found.")
        ci_df[point_col] = ci_df[metric]
        ci_df[lo_col] = np.nan
        ci_df[hi_col] = np.nan

    id_cols = [c for c in ["model_name", "prompt_id"] if c in ci_df.columns]
    ci_df = ci_df.sort_values(point_col, ascending=True).reset_index(drop=True)

    labels = ci_df[id_cols].apply(
        lambda r: " / ".join(_short_name(str(v)) for v in r), axis=1
    )
    y = np.arange(len(ci_df))
    values = ci_df[point_col].values

    fig, ax = plt.subplots(figsize=figsize)

    # Horizontal lines from 0 to point estimate
    for i, (yi, val) in enumerate(zip(y, values)):
        color = list(PROMPT_COLORS.values())[i % len(PROMPT_COLORS)]
        ax.plot([0, val], [yi, yi], color=color, linewidth=1.5, alpha=0.5)
        ax.scatter(val, yi, color=color, s=60, zorder=3)

    # CI bars (if available)
    if not ci_df[lo_col].isna().all():
        for i, (yi, lo, hi) in enumerate(
            zip(y, ci_df[lo_col].values, ci_df[hi_col].values)
        ):
            if not (np.isnan(lo) or np.isnan(hi)):
                ax.plot([lo, hi], [yi, yi], color="dimgrey", linewidth=2, alpha=0.6,
                        solid_capstyle="round")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(
        f"{metric.replace('_', ' ').title()} with 95% Bootstrap CI", fontsize=12
    )
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axvline(0.70, linestyle="--", color="grey", linewidth=0.8, alpha=0.7)
    ax.text(0.71, len(ci_df) - 0.5, "Clinical\nthreshold", fontsize=7,
            color="grey", va="top")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. ROC curves
# ---------------------------------------------------------------------------


def plot_roc_curves(
    results_df: pd.DataFrame,
    prob_cols: Optional[dict[str, str]] = None,
    group_col: str = "model_name",
    figsize: tuple = (7, 6),
) -> Figure:
    """Macro one-vs-rest ROC curves, one curve per model (or prompt).

    Requires per-class probability columns in ``results_df``.

    Parameters
    ----------
    results_df:
        Raw results DataFrame from ``CHDEvaluator.evaluate``.
    prob_cols:
        Mapping from class label to column name. Defaults to
        ``{"ASD": "prob_ASD", ...}``.
    group_col:
        Column used to group curves (``"model_name"`` or ``"prompt_id"``).

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns None if probability columns are not present.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    if prob_cols is None:
        prob_cols = {label: f"prob_{label}" for label in LABELS}

    missing = [col for col in prob_cols.values() if col not in results_df.columns]
    if missing:
        warnings.warn(
            f"plot_roc_curves requires probability columns; missing: {missing}. "
            "Skipping ROC plot."
        )
        return None

    fig, ax = plt.subplots(figsize=figsize)

    palette = sns.color_palette("tab10")
    for g_idx, (group_key, group_df) in enumerate(
        results_df.groupby(group_col)
    ):
        valid = group_df.dropna(
            subset=list(prob_cols.values()) + ["true_label"]
        )
        if valid.empty:
            continue

        y_bin = label_binarize(valid["true_label"], classes=LABELS)
        y_score = valid[list(prob_cols.values())].values

        tpr_all, fpr_interp = [], np.linspace(0, 1, 200)
        for k in range(len(LABELS)):
            fpr, tpr, _ = roc_curve(y_bin[:, k], y_score[:, k])
            tpr_all.append(np.interp(fpr_interp, fpr, tpr))

        mean_tpr = np.mean(tpr_all, axis=0)
        macro_auc = auc(fpr_interp, mean_tpr)

        color = palette[g_idx % len(palette)]
        ax.plot(
            fpr_interp, mean_tpr,
            label=f"{_short_name(str(group_key))} (AUC={macro_auc:.3f})",
            color=color, linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Macro One-vs-Rest ROC Curves", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Reliability (calibration) diagram
# ---------------------------------------------------------------------------


def plot_reliability_diagram(
    analysis: dict,
    figsize: tuple = (8, 5),
) -> Optional[Figure]:
    """Reliability diagrams (one panel per condition) for calibration assessment.

    Returns None if no calibration data is available.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    calibration = analysis.get("calibration")
    if not calibration:
        warnings.warn("No calibration data available; skipping reliability diagram.")
        return None

    n = len(calibration)
    n_cols = min(n, 3)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(figsize[0] * n_cols / 2, figsize[1]),
                              squeeze=False)

    palette = sns.color_palette("tab10")
    for ax_idx, (key, cal) in enumerate(calibration.items()):
        row_i, col_i = divmod(ax_idx, n_cols)
        ax = axes[row_i][col_i]

        reliability = cal["reliability"]
        bin_mids = [r[0] for r in reliability]
        frac_pos = [r[1] for r in reliability]
        counts = [r[2] for r in reliability]

        color = palette[ax_idx % len(palette)]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect")
        ax.bar(bin_mids, frac_pos, width=0.08, alpha=0.5, color=color,
               edgecolor="white")
        ax.plot(bin_mids, frac_pos, "o-", color=color, markersize=4)

        title = " / ".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        ax.set_title(
            f"{_short_name(title, 25)}\n"
            f"ECE={cal['ece']:.3f}  MCE={cal['mce']:.3f}",
            fontsize=8,
        )
        ax.set_xlabel("Mean predicted confidence", fontsize=8)
        ax.set_ylabel("Fraction correct", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)

    for ax_idx in range(n, n_rows * n_cols):
        row_i, col_i = divmod(ax_idx, n_cols)
        axes[row_i][col_i].set_visible(False)

    fig.suptitle("Reliability Diagrams (Calibration)", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Holistic comparison dashboard (multi-panel)
# ---------------------------------------------------------------------------


def plot_holistic_dashboard(
    analysis: dict,
    results_df: Optional[pd.DataFrame] = None,
    figsize: tuple = (18, 12),
) -> Figure:
    """Comprehensive multi-panel dashboard for at-a-glance model comparison.

    Layout::

        ┌───────────────────┬─────────────────────────┐
        │  Macro-F1 bar     │  Per-class F1 heatmap   │
        │  (all conditions) │  (all conditions)       │
        ├───────────────────┼─────────────────────────┤
        │  Bootstrap CI     │  Best-model confusion   │
        │  lollipop         │  matrix                 │
        └───────────────────┴─────────────────────────┘

    Parameters
    ----------
    analysis:
        Output of :func:`analyze.run_full_analysis`.
    results_df:
        Raw results DataFrame (optional; needed only for ROC panel).
    figsize:
        Overall figure dimensions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

    # ── Panel A: Macro-F1 grouped bar ──────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 0])
    metrics = analysis["metrics"]
    ci_df = analysis.get("bootstrap_cis", pd.DataFrame())

    if "model_name" in metrics.columns and "prompt_id" in metrics.columns:
        models = metrics["model_name"].unique()
        prompts = metrics["prompt_id"].unique()
        n_models, n_prompts = len(models), len(prompts)
        bar_width = 0.8 / n_prompts
        x = np.arange(n_models)

        for p_idx, prompt in enumerate(prompts):
            sub = metrics[metrics["prompt_id"] == prompt].set_index("model_name")
            vals = [float(sub.loc[m, "macro_f1"]) if m in sub.index else 0.0
                    for m in models]
            offsets = x - 0.4 + bar_width * (p_idx + 0.5)
            color = PROMPT_COLORS.get(str(prompt), None)
            ax_bar.bar(offsets, vals, width=bar_width, label=str(prompt),
                       color=color, edgecolor="white", linewidth=0.5)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([_short_name(m, 15) for m in models],
                                rotation=20, ha="right", fontsize=8)
        ax_bar.set_ylabel("Macro-F1", fontsize=9)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax_bar.axhline(0.70, linestyle="--", color="grey",
                       linewidth=0.8, alpha=0.7)
        ax_bar.legend(title="Prompt", fontsize=7, title_fontsize=7)
        ax_bar.set_title("A  Macro-F1 by Model & Prompt", fontsize=10,
                          loc="left", fontweight="bold")
    else:
        # Single-group case
        ax_bar.bar(range(len(metrics)), metrics["macro_f1"].values)
        ax_bar.set_title("A  Macro-F1", fontsize=10, loc="left", fontweight="bold")

    # ── Panel B: Per-class F1 heatmap ─────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0, 1])
    per_class = analysis["per_class_metrics"]
    id_cols = [c for c in ["model_name", "prompt_id"] if c in per_class.columns]
    per_class = per_class.copy()
    per_class["condition"] = per_class[id_cols].apply(
        lambda r: " / ".join(_short_name(str(v), 12) for v in r), axis=1
    )
    pivot = per_class.pivot_table(index="condition", columns="class", values="f1")
    pivot = pivot.reindex(columns=[c for c in LABELS if c in pivot.columns])
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0.0, vmax=1.0, ax=ax_heat,
        linewidths=0.4, linecolor="lightgrey",
        cbar_kws={"label": "F1", "shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax_heat.set_title("B  Per-Class F1 Heatmap", fontsize=10,
                       loc="left", fontweight="bold")
    ax_heat.set_xlabel("CHD Class", fontsize=9)
    ax_heat.set_ylabel("", fontsize=9)
    ax_heat.tick_params(axis="y", labelsize=8, rotation=0)
    ax_heat.tick_params(axis="x", labelsize=9)

    # ── Panel C: Bootstrap CI lollipop ────────────────────────────────────
    ax_ci = fig.add_subplot(gs[1, 0])
    point_col = "macro_f1_point"
    lo_col = "macro_f1_ci_lower"
    hi_col = "macro_f1_ci_upper"

    ci_data = (
        ci_df if (not ci_df.empty and point_col in ci_df.columns)
        else metrics.rename(columns={"macro_f1": point_col}).assign(
            **{lo_col: np.nan, hi_col: np.nan}
        )
    )
    id_cols_ci = [c for c in ["model_name", "prompt_id"] if c in ci_data.columns]
    ci_data = ci_data.sort_values(point_col, ascending=True).reset_index(drop=True)
    ci_labels = ci_data[id_cols_ci].apply(
        lambda r: " / ".join(_short_name(str(v), 12) for v in r), axis=1
    )
    y_pos = np.arange(len(ci_data))
    palette_list = list(PROMPT_COLORS.values())

    for i, (yi, val) in enumerate(zip(y_pos, ci_data[point_col].values)):
        color = palette_list[i % len(palette_list)]
        ax_ci.plot([0, val], [yi, yi], color=color, linewidth=1.5, alpha=0.4)
        ax_ci.scatter(val, yi, color=color, s=50, zorder=3)

    if not ci_data[lo_col].isna().all():
        for i, (yi, lo, hi) in enumerate(
            zip(y_pos, ci_data[lo_col].values, ci_data[hi_col].values)
        ):
            if not (np.isnan(lo) or np.isnan(hi)):
                ax_ci.plot([lo, hi], [yi, yi], color="dimgrey",
                           linewidth=2.5, alpha=0.5, solid_capstyle="round")

    ax_ci.set_yticks(y_pos)
    ax_ci.set_yticklabels(ci_labels, fontsize=8)
    ax_ci.set_xlim(0, 1)
    ax_ci.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_ci.set_xlabel("Macro-F1", fontsize=9)
    ax_ci.axvline(0.70, linestyle="--", color="grey", linewidth=0.8, alpha=0.7)
    ax_ci.set_title("C  Macro-F1 with 95% Bootstrap CI", fontsize=10,
                     loc="left", fontweight="bold")

    # ── Panel D: Best-model confusion matrix ──────────────────────────────
    ax_cm = fig.add_subplot(gs[1, 1])
    cms = analysis["confusion_matrices"]
    if cms:
        # Select the condition with highest macro-F1 for the showcase CM
        if "macro_f1" in metrics.columns:
            id_cols_m = [c for c in ["model_name", "prompt_id"] if c in metrics.columns]
            best_row = metrics.loc[metrics["macro_f1"].idxmax()]
            best_key = tuple(best_row[c] for c in id_cols_m if c in best_row.index)
            best_key = best_key[0] if len(best_key) == 1 else best_key
            cm = cms.get(best_key, next(iter(cms.values())))
        else:
            cm = next(iter(cms.values()))
            best_key = next(iter(cms.keys()))

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums == 0, 0.0, cm / row_sums.astype(float))
        sns.heatmap(
            cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS,
            vmin=0, vmax=1, ax=ax_cm, cbar=True, linewidths=0.5,
            linecolor="lightgrey", annot_kws={"size": 9},
            cbar_kws={"label": "Recall", "shrink": 0.8},
        )
        best_label = (
            " / ".join(_short_name(str(k), 15) for k in best_key)
            if isinstance(best_key, tuple)
            else _short_name(str(best_key), 30)
        )
        ax_cm.set_title(
            f"D  Best Confusion Matrix\n({best_label})",
            fontsize=10, loc="left", fontweight="bold",
        )
        ax_cm.set_xlabel("Predicted", fontsize=9)
        ax_cm.set_ylabel("True", fontsize=9)
        ax_cm.tick_params(labelsize=9)

    fig.suptitle(
        "CHD-CXR VLM Evaluation — Holistic Comparison Dashboard",
        fontsize=14, y=1.01, fontweight="bold",
    )
    return fig


# ---------------------------------------------------------------------------
# Convenience: save all figures
# ---------------------------------------------------------------------------


def save_all_figures(
    analysis: dict,
    output_dir: str = "figures",
    results_df: Optional[pd.DataFrame] = None,
    fmt: str = "pdf",
) -> None:
    """Generate every available figure and save to ``output_dir``.

    Parameters
    ----------
    analysis:
        Output of :func:`analyze.run_full_analysis`.
    output_dir:
        Directory where figures are saved (created if it does not exist).
    results_df:
        Raw results DataFrame (optional; enables the ROC curve figure).
    fmt:
        File format — ``"pdf"`` (default, vector, best for publication),
        ``"png"``, or ``"svg"``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _save(fig: Optional[Figure], name: str) -> None:
        if fig is None:
            return
        path = out / f"{name}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    print(f"Saving figures to {out.resolve()} …")

    # 1. Holistic dashboard
    _save(plot_holistic_dashboard(analysis, results_df=results_df), "dashboard")

    # 2. All confusion matrices (grid)
    _save(plot_all_confusion_matrices(analysis), "confusion_matrices_grid")

    # 3. Individual confusion matrices
    for key, cm in analysis["confusion_matrices"].items():
        safe_key = "_".join(str(k) for k in key).replace("/", "-").replace(" ", "_")
        title = " / ".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        _save(plot_confusion_matrix(cm, title=title), f"cm_{safe_key}")

    # 4. Macro-F1 grouped bar chart
    try:
        _save(plot_macro_f1_bar(analysis, metric="macro_f1"), "macro_f1_bar")
        _save(plot_macro_f1_bar(analysis, metric="accuracy"), "accuracy_bar")
    except (ValueError, KeyError) as e:
        print(f"  Skipped bar chart: {e}")

    # 5. Per-class F1 heatmap (overall)
    try:
        _save(plot_per_class_heatmap(analysis, metric="f1"), "per_class_f1_heatmap")
        _save(plot_per_class_heatmap(analysis, metric="sensitivity"),
              "per_class_sensitivity_heatmap")
    except (ValueError, KeyError) as e:
        print(f"  Skipped heatmap: {e}")

    # 6. Per-prompt heatmaps
    if "prompt_id" in analysis["per_class_metrics"].columns:
        for prompt_id in analysis["per_class_metrics"]["prompt_id"].unique():
            try:
                fig = plot_per_class_heatmap(analysis, metric="f1",
                                              prompt_filter=prompt_id)
                _save(fig, f"per_class_f1_{prompt_id}")
            except (ValueError, KeyError):
                pass

    # 7. Bootstrap CI lollipop
    try:
        _save(plot_bootstrap_ci(analysis, metric="macro_f1"), "bootstrap_ci_macro_f1")
    except (ValueError, KeyError) as e:
        print(f"  Skipped CI plot: {e}")

    # 8. ROC curves (requires probability columns)
    if results_df is not None:
        fig = plot_roc_curves(results_df)
        _save(fig, "roc_curves")

    # 9. Reliability / calibration diagram
    _save(plot_reliability_diagram(analysis), "reliability_diagram")

    print("Done.")
