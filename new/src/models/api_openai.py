"""
OpenAI API Backend (GPT-4o)
============================
Wraps the OpenAI API for vision-language inference.

Image encoding: base64 data URI embedded in the messages content array.
Format: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}

Retry logic: exponential backoff on RateLimitError and APITimeoutError.
Cost tracking: populated from config price fields × response.usage token counts.
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


class OpenAIModel(VLMModel):
    """GPT-4o (and compatible OpenAI vision models).

    Parameters
    ----------
    config:
        :class:`ModelConfig` with ``backend="openai"``.
        ``config.inference_params`` may include: ``temperature``, ``seed``.
    api_key:
        OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
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
        """Instantiate the openai.OpenAI client and validate the key exists."""
        import os
        import openai

        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OpenAI API key not found. Set OPENAI_API_KEY env var or pass "
                "api_key to OpenAIModel."
            )
        self._client = openai.OpenAI(api_key=key)
        logger.info("OpenAI client initialised for model %s.", self.model_id)

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
        """Build the OpenAI image_url content block."""
        b64 = OpenAIModel._encode_image_b64(image)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image, prompt: str) -> PredictResult:
        """Call the OpenAI chat completions API with retry logic."""
        import openai

        if self._client is None:
            self.load()

        messages = [
            {
                "role": "user",
                "content": [
                    self._image_content_block(image),
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        api_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.config.max_new_tokens,
            **self.config.inference_params,
        }

        delay = self._retry_base_delay
        last_error: Optional[str] = None

        for attempt in range(self._max_retries + 1):
            t_start = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**api_kwargs)
                latency = time.perf_counter() - t_start

                raw_output = response.choices[0].message.content or ""
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
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

            except (openai.RateLimitError, openai.APITimeoutError) as exc:
                last_error = str(exc)
                if attempt < self._max_retries:
                    logger.warning(
                        "OpenAI transient error (attempt %d/%d): %s — retrying in %.1fs",
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
        """Sequential predict() calls. Parallelism is handled by EvaluationRunner."""
        return [self.predict(img, prompt) for img, prompt in zip(images, prompts)]
