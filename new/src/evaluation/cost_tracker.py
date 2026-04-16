"""
API Cost Tracker
================
Accumulates and reports API inference costs across an evaluation run.

CostTracker.record() is called once per PredictResult by EvaluationRunner.
CostTracker.summary() returns a tidy DataFrame for reporting in the paper.
CostTracker.from_results_df() reconstructs cost totals from a saved CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CostTracker:
    """Accumulate API inference costs across a full evaluation run.

    Usage::

        tracker = CostTracker()
        # Inside evaluation loop:
        tracker.record(model_name, prompt_id, result.input_tokens,
                       result.output_tokens, result.cost_usd)

        # After run:
        print(tracker.summary())
        print(f"Total cost: ${tracker.total_cost_usd:.4f}")
    """

    _records: list[dict] = field(default_factory=list)

    def record(
        self,
        model_name: str,
        prompt_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record one API call's cost."""
        self._records.append(
            {
                "model_name": model_name,
                "prompt_id": prompt_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
            }
        )

    def summary(self) -> pd.DataFrame:
        """Return total cost per (model_name, prompt_id).

        Returns
        -------
        pd.DataFrame
            Columns: ``model_name``, ``prompt_id``, ``n_calls``,
            ``total_input_tokens``, ``total_output_tokens``, ``total_cost_usd``,
            ``avg_cost_per_sample_usd``.
        """
        if not self._records:
            return pd.DataFrame(
                columns=[
                    "model_name", "prompt_id", "n_calls",
                    "total_input_tokens", "total_output_tokens",
                    "total_cost_usd", "avg_cost_per_sample_usd",
                ]
            )

        df = pd.DataFrame(self._records)
        summary = (
            df.groupby(["model_name", "prompt_id"])
            .agg(
                n_calls=("cost_usd", "count"),
                total_input_tokens=("input_tokens", "sum"),
                total_output_tokens=("output_tokens", "sum"),
                total_cost_usd=("cost_usd", "sum"),
            )
            .reset_index()
        )
        summary["avg_cost_per_sample_usd"] = (
            summary["total_cost_usd"] / summary["n_calls"]
        ).round(6)
        summary["total_cost_usd"] = summary["total_cost_usd"].round(4)
        return summary.sort_values(["model_name", "prompt_id"]).reset_index(drop=True)

    @property
    def total_cost_usd(self) -> float:
        """Total accumulated cost in USD across all recorded calls."""
        if not self._records:
            return 0.0
        return round(sum(r["cost_usd"] for r in self._records), 4)

    @classmethod
    def from_results_df(cls, df: pd.DataFrame) -> "CostTracker":
        """Reconstruct a CostTracker from a saved results DataFrame.

        Parameters
        ----------
        df:
            Results DataFrame with columns ``model_name``, ``prompt_id``,
            ``input_tokens``, ``output_tokens``, ``cost_usd``.

        Returns
        -------
        CostTracker
            Populated with one record per row in ``df`` where cost_usd > 0.
        """
        tracker = cls()
        required = {"model_name", "prompt_id", "input_tokens", "output_tokens", "cost_usd"}
        missing = required - set(df.columns)
        if missing:
            return tracker  # Return empty tracker if columns absent

        for _, row in df.iterrows():
            if row.get("cost_usd", 0.0) > 0:
                tracker.record(
                    model_name=str(row["model_name"]),
                    prompt_id=str(row["prompt_id"]),
                    input_tokens=int(row.get("input_tokens", 0) or 0),
                    output_tokens=int(row.get("output_tokens", 0) or 0),
                    cost_usd=float(row["cost_usd"]),
                )
        return tracker
