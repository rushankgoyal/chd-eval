"""
Evaluation Runner
=================
Orchestrates the full model × prompt experiment matrix.

Responsibilities:
  - Load experiment config (YAML): which models, which prompts, settings
  - For each model: load → run all assigned prompts → unload (VRAM management)
  - Per (model, prompt): call model.predict() / predict_batch() over all samples
  - Parse predicted labels from PredictResult.raw_output
  - Assemble and return the canonical results DataFrame
  - Save partial results to disk after each (model, prompt) run
  - Track cumulative API cost via CostTracker

Usage
-----
    from src.data.dataset import load_samples_from_directory
    from src.evaluation.runner import EvaluationRunner, ExperimentConfig

    samples = load_samples_from_directory("data/chd-cxr")
    config = ExperimentConfig.from_yaml("configs/experiments/hf_full.yaml")
    runner = EvaluationRunner(config, samples)
    df = runner.run()
    df.to_csv("results/combined_hf.csv", index=False)
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from tqdm import tqdm

from src.data.dataset import CHDSample
from src.evaluation.cost_tracker import CostTracker
from src.evaluation.label_parser import parse_predicted_label
from src.evaluation.prompts import PROMPTS
from src.models.base import ModelConfig, PredictResult, VLMModel
from src.models.registry import build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Defines one complete evaluation run.

    Parameters
    ----------
    model_configs:
        List of :class:`ModelConfig` instances (one per model to evaluate).
    prompt_ids:
        List of prompt strategy keys from :data:`PROMPTS`,
        e.g. ``["ZSD", "CoT", "RCE", "RAC"]``.
    results_dir:
        Directory where partial and combined CSV results are written.
    batch_size:
        Batch size for HF models. API models ignore this.
    api_workers:
        Number of concurrent threads for API model inference.
        Keep ≤ 5 to stay within RPM limits on most tiers.
    save_partial:
        If True, write a CSV after each (model, prompt) pair completes.
    max_samples:
        If set, evaluate only the first N samples (useful for smoke tests).
    """

    model_configs: list[ModelConfig]
    prompt_ids: list[str]
    results_dir: Path = field(default_factory=lambda: Path("results"))
    batch_size: int = 4
    api_workers: int = 3
    save_partial: bool = True
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        for pid in self.prompt_ids:
            if pid not in PROMPTS:
                raise ValueError(
                    f"Unknown prompt_id '{pid}'. Valid options: {sorted(PROMPTS)}"
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load an ExperimentConfig from a YAML file.

        The YAML may reference model YAML files by path (relative to the
        experiment file's directory or absolute):

        .. code-block:: yaml

            results_dir: results/hf_full
            batch_size: 4
            api_workers: 3
            save_partial: true
            models:
              - configs/models/medgemma_4b.yaml
              - configs/models/gpt4o.yaml
            prompts:
              - ZSD
              - CoT

        Alternatively ``models`` may contain inline dicts with ModelConfig fields.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        base_dir = path.parent
        model_configs: list[ModelConfig] = []
        for entry in data.get("models", []):
            if isinstance(entry, str):
                # Resolve relative to experiment YAML location
                model_path = Path(entry) if Path(entry).is_absolute() else base_dir / entry
                model_configs.append(ModelConfig.from_yaml(model_path))
            elif isinstance(entry, dict):
                model_configs.append(ModelConfig(**entry))
            else:
                raise ValueError(f"Unexpected model entry type: {type(entry)}")

        return cls(
            model_configs=model_configs,
            prompt_ids=data.get("prompts", list(PROMPTS.keys())),
            results_dir=Path(data.get("results_dir", "results")),
            batch_size=int(data.get("batch_size", 4)),
            api_workers=int(data.get("api_workers", 3)),
            save_partial=bool(data.get("save_partial", True)),
            max_samples=data.get("max_samples"),
        )


# ---------------------------------------------------------------------------
# EvaluationRunner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Drives the full model × prompt experiment matrix.

    Parameters
    ----------
    experiment_config:
        :class:`ExperimentConfig` describing the full run.
    samples:
        Pre-loaded list of :class:`CHDSample` instances.
    api_keys:
        Dict mapping backend name to API key string:
        ``{"openai": "sk-...", "anthropic": "sk-ant-...", "google": "AIza..."}``.
        Falls back to environment variables (OPENAI_API_KEY, etc.) if absent.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        samples: list[CHDSample],
        api_keys: Optional[dict[str, str]] = None,
    ) -> None:
        self.config = experiment_config
        self.samples = samples
        self.api_keys = api_keys or {}
        self.cost_tracker = CostTracker()
        self._all_results: list[pd.DataFrame] = []

    def run(self) -> pd.DataFrame:
        """Execute the full experiment matrix.

        Returns
        -------
        pd.DataFrame
            Canonical results DataFrame (one row per (sample, model, prompt)).
            Also saves a combined CSV to ``results_dir/combined.csv``.
        """
        samples = self.samples
        if self.config.max_samples is not None:
            samples = samples[: self.config.max_samples]
            logger.info(
                "max_samples=%d: evaluating %d of %d total samples.",
                self.config.max_samples, len(samples), len(self.samples),
            )

        self.config.results_dir.mkdir(parents=True, exist_ok=True)

        for model_cfg in self.config.model_configs:
            model_kwargs = self._api_kwargs(model_cfg.backend)
            model = build_model(model_cfg, **model_kwargs)

            with model:
                for prompt_id in self.config.prompt_ids:
                    logger.info(
                        "Running %s / %s (%d samples) …",
                        model.display_name, prompt_id, len(samples),
                    )
                    partial_df = self._run_one_condition(model, prompt_id, samples)
                    self._all_results.append(partial_df)

                    if self.config.save_partial:
                        safe_name = model.display_name.replace("/", "-").replace(" ", "_")
                        fname = self.config.results_dir / f"{safe_name}_{prompt_id}.csv"
                        partial_df.to_csv(fname, index=False)
                        logger.info("  Partial results saved to %s", fname)

                    # Update cost tracker
                    for _, row in partial_df.iterrows():
                        if row.get("cost_usd", 0.0) > 0:
                            self.cost_tracker.record(
                                model_name=str(row["model_name"]),
                                prompt_id=str(row["prompt_id"]),
                                input_tokens=int(row.get("input_tokens", 0) or 0),
                                output_tokens=int(row.get("output_tokens", 0) or 0),
                                cost_usd=float(row["cost_usd"]),
                            )

        combined = pd.concat(self._all_results, ignore_index=True)
        combined_path = self.config.results_dir / "combined.csv"
        combined.to_csv(combined_path, index=False)
        logger.info("Combined results saved to %s (%d rows).", combined_path, len(combined))

        # Print parse rate summary
        parse_summary = (
            combined.groupby(["model_name", "prompt_id"])["parse_success"]
            .mean()
            .reset_index()
            .rename(columns={"parse_success": "parse_rate"})
        )
        parse_summary["parse_rate"] = parse_summary["parse_rate"].map("{:.1%}".format)
        print("\nParse rate by condition:")
        print(parse_summary.to_string(index=False))

        if self.cost_tracker.total_cost_usd > 0:
            print(f"\nTotal API cost: ${self.cost_tracker.total_cost_usd:.4f}")
            print(self.cost_tracker.summary().to_string(index=False))

        return combined

    def _run_one_condition(
        self,
        model: VLMModel,
        prompt_id: str,
        samples: list[CHDSample],
    ) -> pd.DataFrame:
        """Run one (model, prompt) pair over all samples and return a DataFrame."""
        prompt_text = PROMPTS[prompt_id]

        if model.is_api_model:
            results = self._run_api_parallel(model, samples, prompt_text)
        else:
            results = self._run_hf_sequential(model, samples, prompt_text)

        return self._build_dataframe(results, model, prompt_id)

    def _run_hf_sequential(
        self,
        model: VLMModel,
        samples: list[CHDSample],
        prompt_text: str,
    ) -> list[tuple[CHDSample, PredictResult]]:
        """Run HF model inference sample-by-sample with a progress bar."""
        pairs: list[tuple[CHDSample, PredictResult]] = []
        for sample in tqdm(samples, desc=f"{model.display_name[:25]} (HF)"):
            result = model.predict(sample.get_image(), prompt_text)
            pairs.append((sample, result))
        return pairs

    def _run_api_parallel(
        self,
        model: VLMModel,
        samples: list[CHDSample],
        prompt_text: str,
    ) -> list[tuple[CHDSample, PredictResult]]:
        """Run API model inference with ThreadPoolExecutor for parallelism."""
        results: list[Optional[tuple[CHDSample, PredictResult]]] = [None] * len(samples)

        def _infer(idx: int, sample: CHDSample):
            result = model.predict(sample.get_image(), prompt_text)
            return idx, sample, result

        with ThreadPoolExecutor(max_workers=self.config.api_workers) as executor:
            futures = {
                executor.submit(_infer, i, s): i
                for i, s in enumerate(samples)
            }
            with tqdm(total=len(samples), desc=f"{model.display_name[:25]} (API)") as pbar:
                for future in as_completed(futures):
                    idx, sample, result = future.result()
                    results[idx] = (sample, result)
                    pbar.update(1)

        return results  # type: ignore[return-value]

    def _build_dataframe(
        self,
        pairs: list[tuple[CHDSample, PredictResult]],
        model: VLMModel,
        prompt_id: str,
    ) -> pd.DataFrame:
        """Convert (CHDSample, PredictResult) pairs into the canonical DataFrame."""
        rows = []
        for sample, result in pairs:
            predicted_label = parse_predicted_label(result.raw_output)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "true_label": sample.label,
                    "predicted_label": predicted_label,
                    "raw_output": result.raw_output,
                    "parse_success": predicted_label is not None,
                    "model_name": model.display_name,
                    "model_id": model.model_id,
                    "backend": model.backend,
                    "prompt_id": prompt_id,
                    "inference_time_s": result.latency_s,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cost_usd": result.cost_usd,
                    "error": result.error,
                }
            )
        df = pd.DataFrame(rows)
        n_failed = int((~df["parse_success"]).sum())
        if n_failed:
            logger.warning(
                "%s / %s: %d / %d samples failed label parsing (%.1f%%).",
                model.display_name, prompt_id,
                n_failed, len(df), 100 * n_failed / len(df),
            )
        return df

    def _api_kwargs(self, backend: str) -> dict:
        """Build API key kwargs for a given backend."""
        mapping = {
            "openai": ("api_key", self.api_keys.get("openai") or os.environ.get("OPENAI_API_KEY")),
            "anthropic": ("api_key", self.api_keys.get("anthropic") or os.environ.get("ANTHROPIC_API_KEY")),
            "gemini": ("api_key", self.api_keys.get("google") or os.environ.get("GOOGLE_API_KEY")),
        }
        if backend in mapping:
            key_name, key_value = mapping[backend]
            if key_value:
                return {key_name: key_value}
        return {}
