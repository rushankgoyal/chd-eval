"""
Anthropic API Backend (Claude)
================================
Wraps the Anthropic API for vision-language inference.

Image encoding: base64, passed as:
    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "<b64>"}}

Retry logic: exponential backoff on anthropic.RateLimitError.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Optional

from PIL import Image

from .base import ModelConfig, PredictResult, VLMModel

logger = logging.getLogger(__name__)


class AnthropicModel(VLMModel):
    """Claude 3.5/3.7 Sonnet (and compatible Anthropic vision models).

    Parameters
    ----------
    config:
        :class:`ModelConfig` with ``backend="anthropic"``.
    api_key:
        Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    max_retries:
        Number of retry attempts on transient API errors.
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
        self._client = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the anthropic.Anthropic client."""
        import os
        import anthropic

        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY env var or pass "
                "api_key to AnthropicModel."
            )
        self._client = anthropic.Anthropic(api_key=key)
        logger.info("Anthropic client initialised for model %s.", self.model_id)

    def unload(self) -> None:
        """No-op for API models."""
        self._client = None

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image_b64(image: Image.Image, fmt: str = "JPEG") -> str:
        """Convert a PIL Image to a base64 string (in-memory, no disk write)."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _image_content_block(image: Image.Image) -> dict:
        """Build the Anthropic image content block."""
        b64 = AnthropicModel._encode_image_b64(image)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        }

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image, prompt: str) -> PredictResult:
        """Call the Anthropic messages API with retry logic."""
        import anthropic

        if self._client is None:
            self.load()

        content = [
            self._image_content_block(image),
            {"type": "text", "text": prompt},
        ]

        delay = self._retry_base_delay
        last_error: Optional[str] = None

        for attempt in range(self._max_retries + 1):
            t_start = time.perf_counter()
            try:
                response = self._client.messages.create(
                    model=self.model_id,
                    max_tokens=self.config.max_new_tokens,
                    messages=[{"role": "user", "content": content}],
                    **self.config.inference_params,
                )
                latency = time.perf_counter() - t_start

                raw_output = response.content[0].text if response.content else ""
                input_tokens = response.usage.input_tokens if response.usage else 0
                output_tokens = response.usage.output_tokens if response.usage else 0
                cost_usd = (
                    input_tokens / 1000 * self.config.cost_per_1k_input_tokens
                    + output_tokens / 1000 * self.config.cost_per_1k_output_tokens
                )

                return PredictResult(
                    raw_output=raw_output.strip(),
                    latency_s=round(latency, 3),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=round(cost_usd, 6),
                    error=None,
                    model_id=self.model_id,
                )

            except anthropic.RateLimitError as exc:
                last_error = str(exc)
                if attempt < self._max_retries:
                    logger.warning(
                        "Anthropic rate limit (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, self._max_retries, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
            except Exception as exc:
                latency = time.perf_counter() - t_start
                return PredictResult(
                    raw_output="",
                    latency_s=round(latency, 3),
                    error=str(exc),
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
