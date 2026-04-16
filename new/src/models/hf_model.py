"""
HuggingFace VLM Backends
=========================
Three concrete subclasses handle the model-specific differences in processor
calls and chat template formats:

    MedGemmaModel   — AutoModelForImageTextToText + apply_chat_template
    LLavaMedModel   — LlavaForConditionalGeneration + <image> token injection
    Qwen25VLModel   — Qwen2_5_VLForConditionalGeneration + process_vision_info

All share the HFVLMModel base class which provides:
  - load() / unload() with proper CUDA memory cleanup
  - predict() with timing, error handling, and token counting
  - predict_batch() with configurable batch_size (chunked to avoid OOM)
"""

from __future__ import annotations

import abc
import gc
import logging
import time
from typing import Any, Optional

import torch
from PIL import Image

from .base import ModelConfig, PredictResult, VLMModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HFVLMModel — shared base for all HuggingFace models
# ---------------------------------------------------------------------------


class HFVLMModel(VLMModel):
    """Abstract base class for HuggingFace-hosted VLMs.

    Subclasses must implement:
        _load_processor_and_model() -> (processor, model)
        _build_inputs(image, prompt) -> dict of tensors on the correct device

    Subclasses may optionally override:
        _decode_output(output_ids, input_len) -> str
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.processor: Any = None
        self.model: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Abstract helpers — subclass responsibility
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _load_processor_and_model(self) -> tuple[Any, Any]:
        """Download and instantiate processor + model.

        Must respect ``config.hf_load_in_4bit`` and ``config.hf_device``.
        Returns ``(processor, model)``.
        """
        ...

    @abc.abstractmethod
    def _build_inputs(self, image: Image.Image, prompt: str) -> dict:
        """Tokenise one (image, prompt) pair into a dict of tensors on device."""
        ...

    def _decode_output(self, output_ids: torch.Tensor, input_len: int) -> str:
        """Decode generated token IDs into a string, stripping the input echo.

        Default implementation uses processor.batch_decode with
        skip_special_tokens=True and strips the result.
        Subclasses may override for model-specific token handling.
        """
        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    # ------------------------------------------------------------------
    # Shared lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load processor and model weights; set model to eval mode."""
        if self._loaded:
            return
        logger.info("Loading %s …", self.display_name)
        self.processor, self.model = self._load_processor_and_model()
        self.model.eval()
        self._loaded = True
        try:
            device_str = str(next(self.model.parameters()).device)
        except StopIteration:
            device_str = "unknown"
        logger.info("%s ready on %s.", self.display_name, device_str)

    def unload(self) -> None:
        """Delete model and processor, then clear CUDA cache."""
        logger.info("Unloading %s …", self.display_name)
        del self.model, self.processor
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_device(self) -> torch.device:
        """Return the device of the first model parameter."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _get_quant_kwargs(self) -> dict:
        """Build quantisation kwargs for from_pretrained if 4-bit is requested."""
        if not self.config.hf_load_in_4bit:
            return {}
        try:
            from transformers import BitsAndBytesConfig
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
        except ImportError:
            logger.warning(
                "bitsandbytes not installed; ignoring hf_load_in_4bit=True. "
                "Install with: pip install bitsandbytes"
            )
            return {}

    # ------------------------------------------------------------------
    # Shared inference
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image, prompt: str) -> PredictResult:
        """Single-sample inference with timing and error capture."""
        t_start = time.perf_counter()
        raw_output = ""
        error_msg = None
        input_tokens = 0
        output_tokens = 0

        try:
            inputs = self._build_inputs(image, prompt)
            input_len = inputs["input_ids"].shape[1]
            input_tokens = input_len

            gen_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                **self.config.inference_params,
            }
            # Ensure greedy decoding by default (reproducible results)
            gen_kwargs.setdefault("do_sample", False)
            if not gen_kwargs.get("do_sample", True):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)

            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

            raw_output = self._decode_output(output_ids, input_len)
            output_tokens = output_ids.shape[1] - input_len

        except Exception as exc:
            error_msg = str(exc)
            logger.warning(
                "Inference failed for %s: %s", self.display_name, exc
            )

        latency = time.perf_counter() - t_start
        return PredictResult(
            raw_output=raw_output,
            latency_s=round(latency, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
            error=error_msg,
            model_id=self.model_id,
        )

    def predict_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        batch_size: int = 1,
    ) -> list[PredictResult]:
        """Process samples in chunks; falls back to one-by-one with batch_size=1."""
        results: list[PredictResult] = []
        for i in range(0, len(images), batch_size):
            chunk_images = images[i : i + batch_size]
            chunk_prompts = prompts[i : i + batch_size]
            for img, prompt in zip(chunk_images, chunk_prompts):
                results.append(self.predict(img, prompt))
        return results


# ---------------------------------------------------------------------------
# MedGemmaModel
# ---------------------------------------------------------------------------


class MedGemmaModel(HFVLMModel):
    """google/medgemma-4b-it and google/medgemma-27b-it.

    Uses ``AutoModelForImageTextToText`` + ``AutoProcessor``.
    Chat template: messages list with ``apply_chat_template(add_generation_prompt=True)``.
    Inputs cast to bfloat16.
    Requires a HuggingFace token (model is gated) — set ``HF_TOKEN`` env var.
    """

    def _load_processor_and_model(self) -> tuple[Any, Any]:
        import os
        from transformers import AutoModelForImageTextToText, AutoProcessor

        hf_token = os.environ.get("HF_TOKEN")
        processor = AutoProcessor.from_pretrained(
            self.model_id, token=hf_token
        )

        kwargs: dict[str, Any] = {
            "token": hf_token,
            "torch_dtype": torch.bfloat16,
            **self._get_quant_kwargs(),
        }
        if self.config.hf_device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.config.hf_device

        model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        return processor, model

    def _build_inputs(self, image: Image.Image, prompt: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        device = self._get_device()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device, dtype=torch.bfloat16)
        return inputs


# ---------------------------------------------------------------------------
# LLavaMedModel
# ---------------------------------------------------------------------------


class LLavaMedModel(HFVLMModel):
    """microsoft/llava-med-v1.5-mistral-7b (and other LLaVA variants).

    Uses ``LlavaForConditionalGeneration``.
    The prompt must include the ``<image>`` token explicitly.
    The processor has separate ``image_processor`` and ``tokenizer`` components.
    """

    # LLaVA default image token
    _IMAGE_TOKEN = "<image>"

    def _load_processor_and_model(self) -> tuple[Any, Any]:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(self.model_id)

        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            **self._get_quant_kwargs(),
        }
        if self.config.hf_device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.config.hf_device

        model = LlavaForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        return processor, model

    def _build_inputs(self, image: Image.Image, prompt: str) -> dict:
        # Prepend the image token if not already present
        if self._IMAGE_TOKEN not in prompt:
            prompt_with_token = f"{self._IMAGE_TOKEN}\n{prompt}"
        else:
            prompt_with_token = prompt

        device = self._get_device()
        inputs = self.processor(
            text=prompt_with_token,
            images=image,
            return_tensors="pt",
        ).to(device)
        return inputs

    def _decode_output(self, output_ids: torch.Tensor, input_len: int) -> str:
        # LLaVA pads on the left; strip input tokens from the beginning
        generated_ids = output_ids[:, input_len:]
        return self.processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()


# ---------------------------------------------------------------------------
# Qwen25VLModel
# ---------------------------------------------------------------------------


class Qwen25VLModel(HFVLMModel):
    """Qwen/Qwen2.5-VL-7B-Instruct (and 72B variant).

    Uses ``Qwen2_5_VLForConditionalGeneration``.
    Requires the ``qwen_vl_utils.process_vision_info()`` helper.
    ``AutoProcessor`` handles both vision and text tokenisation.
    """

    def _load_processor_and_model(self) -> tuple[Any, Any]:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(self.model_id)

        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            **self._get_quant_kwargs(),
        }
        if self.config.hf_device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.config.hf_device

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        )
        return processor, model

    def _build_inputs(self, image: Image.Image, prompt: str) -> dict:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "qwen_vl_utils is required for Qwen2.5-VL. "
                "Install with: pip install qwen-vl-utils"
            ) from e

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        device = self._get_device()
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(device)
        return inputs
