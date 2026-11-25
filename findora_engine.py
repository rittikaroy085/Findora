"""Findora engine: CLIP (ONNX) image/text embeddings plus a saved image index.

This reuses the same models and logic as build_index.py and search_multiple.py,
packaged so the Streamlit app can call it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor, CLIPTokenizer

BASE = Path(__file__).resolve().parent
VISION_DIR = BASE / "models" / "clip" / "vision"
TEXT_DIR = BASE / "models" / "clip" / "text"
INDEX_PATH = BASE / "image_index.npz"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class Findora:
    """Loads the CLIP models once and turns images / text into embeddings."""

    def __init__(self) -> None:
        providers = ["CPUExecutionProvider"]
        self.vision = ort.InferenceSession(
            str(VISION_DIR / "model.onnx"), providers=providers
        )
        self.text_model = ort.InferenceSession(
            str(TEXT_DIR / "model.onnx"), providers=providers
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            str(TEXT_DIR), local_files_only=True
        )
        self.processor = CLIPImageProcessor.from_pretrained(
            str(VISION_DIR), local_files_only=True
        )

    def embed_image(self, path: Path | str) -> np.ndarray:
        with Image.open(path) as img:
            # exif_transpose fixes sideways phone photos
            image = ImageOps.exif_transpose(img).convert("RGB")
        inputs = self.processor(images=image, return_tensors="np")
        vector = self.vision.run(
            ["image_embeds"], {"pixel_values": inputs["pixel_values"]}
        )[0][0]
        return (vector / np.linalg.norm(vector)).astype(np.float32)

    def embed_text(self, query: str) -> np.ndarray:
        tokens = self.tokenizer(
            query,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )
        vector = self.text_model.run(
            ["text_embeds"],
            {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            },
        )[0][0]
        return (vector / np.linalg.norm(vector)).astype(np.float32)

    def search(
        self, query: str, vectors: np.ndarray, top_k: int = 12
    ) -> list[tuple[int, float]]:
        """Return (image position, score) pairs, best match first."""
        scores = vectors @ self.embed_text(query)
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]


@dataclass
class Index:
    folder: str
    paths: list[str]
    vectors: np.ndarray
    mtimes: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def save_index(index: Index) -> None:
    np.savez(
        INDEX_PATH,
        vectors=index.vectors,
        paths=np.array(index.paths),
        mtimes=index.mtimes,
        folder=np.array(index.folder),
    )


def load_index() -> Optional[Index]:
    """Load image_index.npz (also works with the older format from build_index.py)."""
    if not INDEX_PATH.exists():
        return None
    try:
        data = np.load(INDEX_PATH, allow_pickle=False)
        paths = [str(p) for p in data["paths"]]
        vectors = data["vectors"].astype(np.float32)
        if not paths:
            return None
        if "mtimes" in data.files:
            mtimes = data["mtimes"].astype(np.float64)
        else:
            mtimes = np.zeros(len(paths), dtype=np.float64)
        if "folder" in data.files:
            folder = str(data["folder"])
        else:
            folder = os.path.commonpath(paths)
        return Index(folder, paths, vectors, mtimes)
    except Exception:
        return None


def build_index(
    engine: Findora,
    folder: Path | str,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[Index, dict]:
    """Index every image in `folder`, reusing embeddings for unchanged files."""
    folder = Path(folder)
    images = list_images(folder)
    if not images:
        raise ValueError("No supported images (jpg, png, webp, bmp) found in that folder.")

    cache: dict[str, tuple[np.ndarray, float]] = {}
    previous = load_index()
    if previous is not None:
        for p, v, m in zip(previous.paths, previous.vectors, previous.mtimes):
            cache[p] = (v, float(m))

    paths: list[str] = []
    vectors: list[np.ndarray] = []
    mtimes: list[float] = []
    reused = skipped = 0
    total = len(images)

    for n, path in enumerate(images, 1):
        key = str(path)
        try:
            mtime = path.stat().st_mtime
            hit = cache.get(key)
            if hit is not None and hit[1] == mtime:
                vector = hit[0]
                reused += 1
            else:
                vector = engine.embed_image(path)
            paths.append(key)
            vectors.append(vector)
            mtimes.append(mtime)
        except Exception:
            skipped += 1
        if progress:
            progress(n, total, path.name)

    if not vectors:
        raise ValueError("None of the images in that folder could be processed.")

    index = Index(
        folder=str(folder),
        paths=paths,
        vectors=np.stack(vectors).astype(np.float32),
        mtimes=np.array(mtimes, dtype=np.float64),
    )
    save_index(index)
    stats = {
        "total": total,
        "new": len(paths) - reused,
        "reused": reused,
        "skipped": skipped,
    }
    return index, stats
