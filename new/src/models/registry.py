"""
Model Registry
==============
Central registry mapping backend strings to VLMModel subclasses.
``build_model()`` is the single entry point used by EvaluationRunner —
callers never import subclasses directly.
"""

from __future__ import annotations

from .api_anthropic import AnthropicModel
from .api_gemini import GeminiModel
from .api_openai import OpenAIModel
from .base import ModelConfig, VLMModel
from .hf_model import LLavaMedModel, MedGemmaModel, Qwen25VLModel

#: Maps backend string → VLMModel subclass.
MODEL_REGISTRY: dict[str, type[VLMModel]] = {
    "hf_medgemma": MedGemmaModel,
    "hf_llava_med": LLavaMedModel,
    "hf_qwen25_vl": Qwen25VLModel,
    "openai": OpenAIModel,
    "anthropic": AnthropicModel,
    "gemini": GeminiModel,
}


def build_model(config: ModelConfig, **kwargs) -> VLMModel:
    """Instantiate the correct VLMModel subclass for the given config.

    Parameters
    ----------
    config:
        :class:`ModelConfig` loaded from YAML.
    **kwargs:
        Passed through to the subclass constructor.
        For API models: ``api_key``, ``max_retries``, ``retry_base_delay_s``.
        For HF models: these are ignored (HF uses env vars / config fields).

    Returns
    -------
    VLMModel
        An uninitialised instance (call ``load()`` or use as context manager).

    Raises
    ------
    ValueError
        If ``config.backend`` is not in :data:`MODEL_REGISTRY`.
    """
    cls = MODEL_REGISTRY.get(config.backend)
    if cls is None:
        raise ValueError(
            f"Unknown backend '{config.backend}'. "
            f"Valid options: {sorted(MODEL_REGISTRY)}"
        )
    return cls(config, **kwargs)
