"""
Google Gemini API Backend
==========================
Wraps the Google Generative AI (genai) SDK for vision-language inference.

Key difference from OpenAI/Anthropic: Gemini accepts PIL Image objects
directly in the content list — no base64 encoding required.

Token counts come from ``response.usage_metadata``.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from PIL import Image

from .base import ModelConfig, PredictResult, VLMModel

logger = logging.getLogger(__name__)


class GeminiModel(VLMModel):
    """Gemini 1.5 Pro / 2.0 Flash (and compatible Google Generative AI models).

    Parameters
    ----------
    config:
        :class:`ModelConfig` with ``backend="gemini"``.
        ``config.inference_params`` may include: ``temperature``, ``top_p``, ``top_k``.
    api_key:
        Google AI API key. Falls back to ``GOOGLE_API_KEY`` env var.
    max_retries:
        Number of retry attempts on transient errors.
    retry_base_delay_s:
        Initial backoff delay in seconds (doubled each retry).
    """

    def __init__(
        self,
        config: ModelConfig,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_base_delay_s: float = 2.0,
    ) -> None:
        super().__init__(config)
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay_s
        self._model = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Configure genai with API key and instantiate the GenerativeModel."""
        import os
        import google.generativeai as genai

        key = self._api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError(
                "Google API key not found. Set GOOGLE_API_KEY env var or pass "
                "api_key to GeminiModel."
            )
        genai.configure(api_key=key)

        generation_config = {
            "max_output_tokens": self.config.max_new_tokens,
            **self.config.inference_params,
        }
        self._model = genai.GenerativeModel(
            model_name=self.model_id,
            generation_config=generation_config,
        )
        logger.info("Gemini model %s initialised.", self.model_id)

    def unload(self) -> None:
        """No-op for API models."""
        self._model = None

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image, prompt: str) -> PredictResult:
        """Call the Gemini generate_content API with retry logic.

        Gemini accepts a PIL Image directly in the content list — no encoding
        step needed.
        """
        if self._model is None:
            self.load()

        content = [image.convert("RGB"), prompt]

        delay = self._retry_base_delay
        last_error: Optional[str] = None

        for attempt in range(self._max_retries + 1):
            t_start = time.perf_counter()
            try:
                response = self._model.generate_content(content)
                latency = time.perf_counter() - t_start

                raw_output = response.text if hasattr(response, "text") else ""

                # Token counts from usage_metadata (available in most Gemini responses)
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    input_tokens = getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    ) or 0
                    output_tokens = getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ) or 0

                cost_usd = (
                    input_tokens / 1000 * self.config.cost_per_1k_input_tokens
                    + output_tokens / 1000 * self.config.cost_per_1k_output_tokens
                )

                return PredictResult(
                    raw_output=(raw_output or "").strip(),
                    latency_s=round(latency, 3),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=round(cost_usd, 6),
                    error=None,
                    model_id=self.model_id,
                )

            except Exception as exc:
                last_error = str(exc)
                # Retry on likely transient errors (rate limit, timeout)
                error_lower = last_error.lower()
                is_transient = any(
                    kw in error_lower
                    for kw in ("rate", "quota", "timeout", "503", "429", "resource")
                )
                if is_transient and attempt < self._max_retries:
                    latency = time.perf_counter() - t_start
                    logger.warning(
                        "Gemini transient error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, self._max_retries, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    latency = time.perf_counter() - t_start
                    return PredictResult(
                        raw_output="",
                        latency_s=round(latency, 3),
                        error=last_error,
                        model_id=self.model_id,
                    )

        return PredictResult(
            raw_output="",
            latency_s=0.0,
            error=f"Max retries exceeded: {last_error}",
            model_id=self.model_id,
        )

    def predict_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
    ) -> list[PredictResult]:
        """Sequential predict() calls. Parallelism handled by EvaluationRunner."""
        return [self.predict(img, prompt) for img, prompt in zip(images, prompts)]
