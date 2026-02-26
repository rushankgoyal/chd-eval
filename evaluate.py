"""
CHD-CXR VLM Evaluation Pipeline
================================
Standardized pipeline for evaluating HuggingFace vision-language models on
congenital heart disease (CHD) classification from pediatric chest X-rays.

The pipeline accepts a HuggingFace model name, a prompt string, and a list of
CHDSample instances (image + ground-truth label), runs inference, parses the
predicted label from the raw text output, and returns a tidy pandas DataFrame.

Supports the four prompting paradigms from the CHD-VLM research plan:
  P1 – Zero-Shot Direct (ZSD)
  P2 – Role-Conditioned Expert (RCE)
  P3 – Chain-of-Thought Medical Reasoning (CoT)
  P4 – Reference-Anchored Classification (RAC)

Usage
-----
    import os
    from evaluate import CHDEvaluator, CHDSample, PROMPTS, load_samples_from_directory

    samples = load_samples_from_directory("data/chd-cxr")
    evaluator = CHDEvaluator("google/medgemma-4b-it", hf_token=os.environ["HF_TOKEN"])
    df = evaluator.evaluate(samples, prompt=PROMPTS["ZSD"], prompt_id="ZSD")
    df.to_csv("results_medgemma_zsd.csv", index=False)

HF Token
--------
Set the HF_TOKEN environment variable before running:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
Or pass it directly as the hf_token argument to CHDEvaluator.
Gated models (e.g. MedGemma) additionally require accepting the model licence
on the HuggingFace website with the same account.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELS: list[str] = ["ASD", "VSD", "PDA", "Normal"]

# Canonical long-form names and common synonyms that models may produce.
LABEL_ALIASES: dict[str, list[str]] = {
    "ASD": [
        "asd",
        "atrial septal defect",
        "atrial septal",
        "ostium secundum",
        "secundum asd",
        "right heart enlargement",
    ],
    "VSD": [
        "vsd",
        "ventricular septal defect",
        "ventricular septal",
        "interventricular communication",
        "interventricular defect",
    ],
    "PDA": [
        "pda",
        "patent ductus arteriosus",
        "ductus arteriosus",
        "ductal",
        "patent duct",
    ],
    "Normal": [
        "normal",
        "no chd",
        "no congenital",
        "no defect",
        "no abnormality",
        "healthy",
        "unremarkable",
        "within normal limits",
        "no significant abnormality",
    ],
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class CHDSample:
    """A single labelled pediatric chest X-ray sample.

    Parameters
    ----------
    image:
        Either a ``PIL.Image.Image`` object or a file path (str / Path).
        Paths are resolved lazily on first access via ``get_image()``.
    label:
        Ground-truth class label. Must be one of ``LABELS``
        (``"ASD"``, ``"VSD"``, ``"PDA"``, ``"Normal"``).
    sample_id:
        Optional identifier (e.g. filename stem or dataset row index).
        Recorded in the results DataFrame.
    image_path:
        Optional string representation of the source path, used for logging
        and the results DataFrame. Inferred automatically when ``image`` is a
        path string.
    """

    image: Union[Image.Image, str, Path]
    label: str
    sample_id: Optional[str] = None
    image_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(
                f"label must be one of {LABELS}, got '{self.label}'. "
                "Use load_samples_from_directory(label_map=...) to remap custom names."
            )
        if isinstance(self.image, (str, Path)) and self.image_path is None:
            self.image_path = str(self.image)

    def get_image(self) -> Image.Image:
        """Return a PIL RGB Image, loading from disk if necessary."""
        if isinstance(self.image, Image.Image):
            return self.image.convert("RGB")
        return Image.open(self.image).convert("RGB")


# ---------------------------------------------------------------------------
# Prompting paradigms
# ---------------------------------------------------------------------------

#: Four prompting strategies as defined in the CHD-VLM research plan.
#: Each value is a ready-to-use prompt string. Pass the desired value to
#: ``CHDEvaluator.evaluate()`` along with the matching ``prompt_id`` key.
PROMPTS: dict[str, str] = {
    # ------------------------------------------------------------------
    # P1 – Zero-Shot Direct (ZSD)
    # Baseline: minimal framing, forces a bare label response.
    # ------------------------------------------------------------------
    "ZSD": (
        "You are a medical image analysis system. "
        "Examine this pediatric chest X-ray and classify it into exactly one of the "
        "following four categories:\n"
        "  • ASD (Atrial Septal Defect)\n"
        "  • VSD (Ventricular Septal Defect)\n"
        "  • PDA (Patent Ductus Arteriosus)\n"
        "  • Normal\n\n"
        "Reply with only the category label (ASD, VSD, PDA, or Normal) and nothing else."
    ),

    # ------------------------------------------------------------------
    # P2 – Role-Conditioned Expert (RCE)
    # Expert persona conditioning; expected to boost accuracy ~8–12%.
    # ------------------------------------------------------------------
    "RCE": (
        "You are an expert pediatric cardiologist and radiologist with 20 years of "
        "experience interpreting neonatal and pediatric chest X-rays. "
        "You are reviewing a chest radiograph from a pediatric patient for possible "
        "congenital heart disease (CHD).\n\n"
        "Classify this image into exactly one of:\n"
        "  • ASD (Atrial Septal Defect)\n"
        "  • VSD (Ventricular Septal Defect)\n"
        "  • PDA (Patent Ductus Arteriosus)\n"
        "  • Normal\n\n"
        "Reply with only the category label (ASD, VSD, PDA, or Normal) and nothing else."
    ),

    # ------------------------------------------------------------------
    # P3 – Chain-of-Thought Medical Reasoning (CoT)
    # Step-by-step pathophysiological reasoning before final diagnosis.
    # Parser looks for the terminal "DIAGNOSIS: <label>" line.
    # ------------------------------------------------------------------
    "CoT": (
        "You are an expert pediatric cardiologist. "
        "Examine this chest X-ray step by step using the following framework:\n\n"
        "Step 1 – Cardiac silhouette: Assess size (cardiothoracic ratio), shape, and borders.\n"
        "Step 2 – Pulmonary vascularity: Describe pulmonary blood flow "
        "(increased, normal, or decreased).\n"
        "Step 3 – Mediastinal structures: Evaluate the aortic knuckle, superior "
        "mediastinum width, and tracheal position.\n"
        "Step 4 – Bony and parenchymal findings: Note rib notching, lung fields, "
        "and pleural spaces.\n"
        "Step 5 – Integration: Synthesise findings into a single diagnosis.\n\n"
        "After your reasoning, output your final answer on the last line in the exact format:\n"
        "DIAGNOSIS: <label>\n"
        "where <label> is one of: ASD, VSD, PDA, Normal."
    ),

    # ------------------------------------------------------------------
    # P4 – Reference-Anchored Classification (RAC)
    # Canonical per-class radiographic features provided as textual anchors.
    # ------------------------------------------------------------------
    "RAC": (
        "You are an expert pediatric cardiologist. "
        "Classify this pediatric chest X-ray into one of the four categories below, "
        "using the canonical radiographic features as anchors:\n\n"
        "ASD (Atrial Septal Defect):\n"
        "  – Mild to moderate cardiomegaly with right heart enlargement\n"
        "  – Increased pulmonary vascularity (shunt vascularity / plethora)\n"
        "  – Prominent main pulmonary artery segment\n\n"
        "VSD (Ventricular Septal Defect):\n"
        "  – Cardiomegaly proportional to shunt size\n"
        "  – Increased pulmonary vascularity (biventricular volume overload pattern)\n"
        "  – Enlarged left atrium and left ventricle\n\n"
        "PDA (Patent Ductus Arteriosus):\n"
        "  – Cardiomegaly with left heart prominence\n"
        "  – Pulmonary oedema or markedly increased pulmonary flow\n"
        "  – Prominent aortic arch / knuckle\n\n"
        "Normal:\n"
        "  – Normal cardiothoracic ratio (< 0.5)\n"
        "  – Normal pulmonary vascularity\n"
        "  – Clear lung fields, no mediastinal widening\n\n"
        "Reply with only the category label (ASD, VSD, PDA, or Normal) and nothing else."
    ),
}


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def parse_predicted_label(text: str) -> Optional[str]:
    """Extract a CHD class label from raw model output text.

    Applies the following strategies in order, returning on first success:

    1. The entire response (stripped) exactly matches a short label.
    2. ``DIAGNOSIS: <label>`` pattern (CoT prompt format).
    3. JSON object containing a ``"diagnosis"`` / ``"label"`` / similar key.
    4. Long-form synonym or alias match anywhere in the text.
    5. Last bare occurrence of ASD / VSD / PDA / Normal in the text.

    Returns ``None`` if no label can be extracted.
    """
    if not text:
        return None

    stripped = text.strip()

    # 1. Exact match against the entire stripped response
    for label in LABELS:
        if stripped.lower() == label.lower():
            return label

    # 2. "DIAGNOSIS: <label>" pattern (CoT output format)
    m = re.search(r"DIAGNOSIS\s*:\s*(\w[\w\s]*)", stripped, re.IGNORECASE)
    if m:
        matched = _match_alias(m.group(1).strip())
        if matched:
            return matched

    # 3. JSON structure (models that produce structured output)
    json_match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            for key in ("diagnosis", "label", "classification", "prediction", "class", "result"):
                if key in obj:
                    matched = _match_alias(str(obj[key]))
                    if matched:
                        return matched
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. Synonym / long-form match anywhere in the text
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", stripped, re.IGNORECASE):
                return label

    # 5. Last bare label occurrence (covers most CoT free-form endings)
    matches = re.findall(r"\b(ASD|VSD|PDA|Normal|NORMAL)\b", stripped, re.IGNORECASE)
    if matches:
        candidate = matches[-1]
        if candidate.upper() in ("ASD", "VSD", "PDA"):
            return candidate.upper()
        if candidate.lower() == "normal":
            return "Normal"

    return None


def _match_alias(text: str) -> Optional[str]:
    """Return the canonical LABEL that ``text`` aliases to, or None."""
    text_lower = text.strip().lower()
    for label, aliases in LABEL_ALIASES.items():
        if text_lower == label.lower() or text_lower in aliases:
            return label
    return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class CHDEvaluator:
    """Evaluate a HuggingFace vision-language model on CHD-CXR classification.

    Loads the model once in ``__init__`` then exposes :meth:`evaluate` which
    iterates over a list of :class:`CHDSample` instances and returns a tidy
    pandas DataFrame suitable for downstream analysis.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. ``"google/medgemma-4b-it"``.
    hf_token:
        HuggingFace access token. Falls back to the ``HF_TOKEN`` environment
        variable if not provided.
    device:
        ``"auto"`` (default) lets ``accelerate`` choose the best device.
        Pass ``"cuda:0"``, ``"cpu"``, etc. to override.
    load_in_4bit:
        Quantise to 4-bit using bitsandbytes (requires GPU + ``bitsandbytes``
        package). Reduces VRAM requirement at a minor accuracy cost.
    max_new_tokens:
        Maximum tokens the model may generate per sample.
        256 is sufficient for label-only prompts; use 512+ for CoT.
    """

    def __init__(
        self,
        model_name: str,
        hf_token: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.max_new_tokens = max_new_tokens
        self._device_arg = device
        self.processor, self.model = self._load_model(load_in_4bit)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, load_in_4bit: bool):
        """Load processor and model from HuggingFace Hub."""
        from transformers import AutoProcessor

        logger.info("Loading processor for %s …", self.model_name)
        processor = AutoProcessor.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True,
        )

        kwargs: dict[str, Any] = {
            "token": self.hf_token,
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed; ignoring load_in_4bit=True. "
                    "Install with: pip install bitsandbytes"
                )

        if self._device_arg == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self._device_arg

        logger.info("Loading model weights for %s …", self.model_name)

        # Try Vision2Seq first (PaliGemma, InstructBLIP, etc.),
        # then fall back to CausalLM (LLaVA, MedGemma, etc.).
        model = None
        for cls_name in ("AutoModelForVision2Seq", "AutoModelForCausalLM"):
            try:
                import transformers
                cls = getattr(transformers, cls_name)
                model = cls.from_pretrained(self.model_name, **kwargs)
                logger.info("Loaded with %s.", cls_name)
                break
            except (AttributeError, ValueError, TypeError, OSError) as exc:
                logger.debug("Could not load with %s: %s", cls_name, exc)

        if model is None:
            raise RuntimeError(
                f"Could not load '{self.model_name}' with AutoModelForVision2Seq "
                "or AutoModelForCausalLM. Check the model ID and your HF token."
            )

        model.eval()
        try:
            device_str = str(next(model.parameters()).device)
        except StopIteration:
            device_str = "unknown"
        logger.info("Model ready on %s.", device_str)
        return processor, model

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _build_inputs(self, image: Image.Image, prompt_text: str) -> dict:
        """Tokenise an (image, prompt) pair into model inputs.

        Uses ``apply_chat_template`` when available (modern instruction-tuned
        models), otherwise falls back to a bare ``<image>\\n<prompt>`` string.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, Exception):
            # Fallback for models without a chat template
            text = f"<image>\n{prompt_text}"

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        )

        # Move all tensors to the model's primary device
        device = next(self.model.parameters()).device
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_sample(self, sample: CHDSample, prompt: str) -> dict:
        """Run inference on a single :class:`CHDSample`.

        Returns a dict with all result fields (see :meth:`evaluate` for the
        full column list).
        """
        image = sample.get_image()
        t_start = time.perf_counter()
        raw_output = ""
        error_msg = None

        try:
            inputs = self._build_inputs(image, prompt)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,     # greedy / deterministic
                    temperature=None,
                    top_p=None,
                )

            # Decode only the newly generated tokens (strip the prompt echo)
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
            raw_output = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

        except Exception as exc:
            error_msg = str(exc)
            logger.warning(
                "Inference failed for sample '%s': %s", sample.sample_id, exc
            )

        inference_time = time.perf_counter() - t_start
        predicted_label = parse_predicted_label(raw_output)

        return {
            "sample_id": sample.sample_id,
            "image_path": sample.image_path,
            "true_label": sample.label,
            "predicted_label": predicted_label,
            "raw_output": raw_output,
            "parse_success": predicted_label is not None,
            "inference_time_s": round(inference_time, 3),
            "error": error_msg,
        }

    def evaluate(
        self,
        samples: list[CHDSample],
        prompt: str,
        prompt_id: str = "custom",
    ) -> pd.DataFrame:
        """Evaluate the model on all samples and return a results DataFrame.

        Parameters
        ----------
        samples:
            List of :class:`CHDSample` instances to evaluate.
        prompt:
            The prompt text to use for every sample. Typically one of the
            values in :data:`PROMPTS`, but any string is accepted.
        prompt_id:
            Short identifier recorded in the ``prompt_id`` column
            (e.g. ``"ZSD"``, ``"CoT"``).

        Returns
        -------
        pd.DataFrame
            One row per sample with columns:

            ==================  =================================================
            sample_id           Identifier from :class:`CHDSample`
            image_path          Source path (or None)
            true_label          Ground-truth class label
            predicted_label     Parsed predicted label (None if parsing failed)
            raw_output          Full unprocessed model output string
            parse_success       Whether label parsing succeeded (bool)
            model_name          Model identifier (``self.model_name``)
            prompt_id           The ``prompt_id`` argument
            inference_time_s    Wall-clock inference time in seconds
            error               Exception message if inference failed (else None)
            ==================  =================================================
        """
        rows = []
        for sample in tqdm(samples, desc=f"{self.model_name[:30]} / {prompt_id}"):
            row = self.evaluate_sample(sample, prompt)
            row["model_name"] = self.model_name
            row["prompt_id"] = prompt_id
            rows.append(row)

        df = pd.DataFrame(rows)

        n_failed = int((~df["parse_success"]).sum())
        if n_failed:
            logger.warning(
                "%d / %d samples failed label parsing (%.1f%%). "
                "Check raw_output column for diagnosis.",
                n_failed,
                len(df),
                100 * n_failed / len(df),
            )
        else:
            logger.info("All %d samples parsed successfully.", len(df))

        return df


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def load_samples_from_directory(
    root: Union[str, Path],
    label_map: Optional[dict[str, str]] = None,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> list[CHDSample]:
    """Load labelled samples from a directory tree structured as::

        root/
            ASD/
                image001.png
                ...
            VSD/
                image002.png
                ...
            PDA/ ...
            Normal/ ...

    Parameters
    ----------
    root:
        Root directory containing one subdirectory per class.
    label_map:
        Optional mapping from subdirectory name to canonical CHD label,
        e.g. ``{"asd": "ASD", "ctrl": "Normal"}``.  If omitted, subdir
        names are matched case-insensitively against :data:`LABELS`.
    extensions:
        Tuple of file extensions to include (lowercase, with leading dot).

    Returns
    -------
    list[CHDSample]
        Samples sorted deterministically by (label, filename).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    samples: list[CHDSample] = []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        raw_name = subdir.name
        # Resolve label via explicit map, then case-insensitive match against LABELS
        label = (label_map or {}).get(raw_name)
        if label is None:
            label = next(
                (l for l in LABELS if l.lower() == raw_name.lower()), None
            )
        if label is None:
            logger.warning(
                "Skipping subdirectory '%s': cannot map to a known label. "
                "Use label_map to handle custom directory names.",
                raw_name,
            )
            continue

        img_paths = sorted(
            p for p in subdir.iterdir()
            if p.suffix.lower() in extensions
        )
        for img_path in img_paths:
            samples.append(
                CHDSample(
                    image=img_path,
                    label=label,
                    sample_id=img_path.stem,
                    image_path=str(img_path),
                )
            )

    logger.info(
        "Loaded %d samples from %s  (%s)",
        len(samples),
        root,
        ", ".join(f"{l}: {sum(s.label == l for s in samples)}" for l in LABELS),
    )
    return samples


def load_samples_from_csv(
    csv_path: Union[str, Path],
    image_col: str = "image_path",
    label_col: str = "label",
    id_col: Optional[str] = "sample_id",
) -> list[CHDSample]:
    """Load samples from a CSV file with image paths and labels.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    image_col:
        Name of the column containing image file paths.
    label_col:
        Name of the column containing class labels (values must be in
        :data:`LABELS`).
    id_col:
        Optional column name for sample identifiers. If the column does not
        exist the row index is used instead.

    Returns
    -------
    list[CHDSample]
    """
    df = pd.read_csv(csv_path)
    samples = []
    for idx, row in df.iterrows():
        sid = str(row[id_col]) if (id_col and id_col in df.columns) else str(idx)
        samples.append(
            CHDSample(
                image=row[image_col],
                label=row[label_col],
                sample_id=sid,
                image_path=str(row[image_col]),
            )
        )
    return samples
