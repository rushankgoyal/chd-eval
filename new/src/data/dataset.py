"""
CHD-CXR Dataset
===============
Data classes and loaders for the Congenital Heart Disease chest X-ray dataset.

Dataset structure expected on disk::

    root/
        ASD/
            image001.png
            ...
        VSD/
            image002.png
        PDA/ ...
        Normal/ ...

Each subdirectory name is matched case-insensitively to the canonical labels
{ASD, VSD, PDA, Normal}. A ``label_map`` can override directory-to-label mapping
for datasets with non-standard subdirectory names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

LABELS: list[str] = ["ASD", "VSD", "PDA", "Normal"]


@dataclass
class CHDSample:
    """A single labelled pediatric chest X-ray sample.

    Parameters
    ----------
    image:
        Either a ``PIL.Image.Image`` object or a file path (str / Path).
        Paths are resolved lazily on first access via ``get_image()``.
    label:
        Ground-truth class label. Must be one of :data:`LABELS`.
    sample_id:
        Optional identifier (e.g. filename stem or dataset row index).
        Recorded in the results DataFrame.
    image_path:
        String representation of the source path, used for logging and
        the results DataFrame. Inferred automatically when ``image`` is a path.
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


def load_samples_from_directory(
    root: Union[str, Path],
    label_map: Optional[dict[str, str]] = None,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> list[CHDSample]:
    """Load labelled samples from a directory tree structured as::

        root/
            ASD/   image001.png ...
            VSD/   image002.png ...
            PDA/   ...
            Normal/ ...

    Parameters
    ----------
    root:
        Root directory containing one subdirectory per class.
    label_map:
        Optional mapping from subdirectory name to canonical CHD label,
        e.g. ``{"asd": "ASD", "ctrl": "Normal"}``. If omitted, subdir
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
        label = (label_map or {}).get(raw_name)
        if label is None:
            label = next((l for l in LABELS if l.lower() == raw_name.lower()), None)
        if label is None:
            logger.warning(
                "Skipping subdirectory '%s': cannot map to a known label. "
                "Use label_map to handle custom directory names.",
                raw_name,
            )
            continue

        img_paths = sorted(p for p in subdir.iterdir() if p.suffix.lower() in extensions)
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
        Column containing image file paths.
    label_col:
        Column containing class labels (values must be in :data:`LABELS`).
    id_col:
        Optional column name for sample identifiers. If absent, row index is used.

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
