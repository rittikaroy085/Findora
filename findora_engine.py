"""Findora engine: CLIP (ONNX) image/text embeddings plus image indexes.

Works in two modes:
  * local  - index a folder on your own PC and save image_index.npz
  * hosted - index photos uploaded by a visitor, kept in memory only
"""

from __future__ import annotations

import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent
VISION_DIR = BASE / "models" / "clip" / "vision"
TEXT_DIR = BASE / "models" / "clip" / "text"
INDEX_PATH = BASE / "image_index.npz"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# The ONNX files are too big for GitHub, so they are fetched from Hugging Face
# the first time the app starts (same files you downloaded by hand).
HF_REPO = "Xenova/clip-vit-base-patch32"
REQUIRED_FILES = {
    VISION_DIR / "model.onnx": "onnx/vision_model.onnx",
    VISION_DIR / "preprocessor_config.json": "preprocessor_config.json",
    TEXT_DIR / "model.onnx": "onnx/text_model.onnx",
    TEXT_DIR / "vocab.json": "vocab.json",
    TEXT_DIR / "merges.txt": "merges.txt",
    TEXT_DIR / "tokenizer.json": "tokenizer.json",
    TEXT_DIR / "tokenizer_config.json": "tokenizer_config.json",
    TEXT_DIR / "special_tokens_map.json": "special_tokens_map.json",
}


def ensure_models(log: Optional[Callable[[str], None]] = None) -> None:
    """Download any missing model file. Files you already have are left alone."""
    missing = [(d, r) for d, r in REQUIRED_FILES.items() if not d.exists()]
    if not missing:
        return
    from huggingface_hub import hf_hub_download

    for dest, remote in missing:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if log:
            log(f"Downloading {dest.parent.name}/{dest.name} ...")
        cached = hf_hub_download(repo_id=HF_REPO, filename=remote)
        shutil.copyfile(cached, dest)


class Findora:
    """Loads the CLIP models once and turns images / text into embeddings."""

    def __init__(self, log: Optional[Callable[[str], None]] = None) -> None:
        ensure_models(log)
        import onnxruntime as ort
        from transformers import CLIPImageProcessor, CLIPTokenizer

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

    def embed_image(self, source) -> np.ndarray:
        """`source` is a file path or a file-like object (e.g. BytesIO)."""
        with Image.open(source) as img:
            img.draft("RGB", (640, 640))  # fast decode for big JPEGs; no-op otherwise
            # exif_transpose fixes sideways phone photos
            image = ImageOps.exif_transpose(img).convert("RGB")
        inputs = self.processor(images=image, return_tensors="np")
        vector = self.vision.run(
            ["image_embeds"], {"pixel_values": inputs["pixel_values"]}
        )[0][0]
        return (vector / np.linalg.norm(vector)).astype(np.float32)

    def embed_text(self, query: str) -> np.ndarray:
        query = query.strip()
        prompts = [query]
        if len(query.split()) <= 4:
            # CLIP understands caption-style text better than single words
            prompts += [f"a photo of {query}", f"a picture of {query}"]
            if len(query.split()) == 1:
                prompts.append(f"a photo of a {query}")
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )
        vectors = self.text_model.run(
            ["text_embeds"],
            {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            },
        )[0]
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        mean = vectors.mean(axis=0)
        return (mean / np.linalg.norm(mean)).astype(np.float32)
    
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
    thumbs: Optional[list] = None  # JPEG bytes per image (upload mode only)

    def __len__(self) -> int:
        return len(self.paths)

    def source(self, i: int):
        """Thumbnail bytes (uploads) or a file path (local folder)."""
        return self.thumbs[i] if self.thumbs is not None else self.paths[i]

    def name(self, i: int) -> str:
        return Path(self.paths[i]).name


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
    save: bool = True,
    use_cache: bool = True,
) -> tuple[Index, dict]:
    """Index every image in `folder`, reusing embeddings for unchanged files."""
    folder = Path(folder)
    images = list_images(folder)
    if not images:
        raise ValueError("No supported images (jpg, png, webp, bmp) found in that folder.")

    cache: dict[str, tuple[np.ndarray, float]] = {}
    previous = load_index() if use_cache else None
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
    if save:
        save_index(index)
    stats = {
        "total": total,
        "new": len(paths) - reused,
        "reused": reused,
        "skipped": skipped,
    }
    return index, stats


def make_thumb(data: bytes, size: int = 480) -> bytes:
    with Image.open(io.BytesIO(data)) as img:
        img.draft("RGB", (size * 2, size * 2))
        thumb = ImageOps.exif_transpose(img).convert("RGB")
    thumb.thumbnail((size, size))
    buf = io.BytesIO()
    thumb.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def build_index_from_uploads(
    engine: Findora,
    files: Sequence[tuple[str, bytes]],
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[Index, dict]:
    """Index uploaded photos entirely in memory. Nothing is written to disk."""
    names: list[str] = []
    vectors: list[np.ndarray] = []
    thumbs: list[bytes] = []
    skipped = 0
    total = len(files)

    for n, (name, data) in enumerate(files, 1):
        try:
            vector = engine.embed_image(io.BytesIO(data))
            thumb = make_thumb(data)
            names.append(name)
            vectors.append(vector)
            thumbs.append(thumb)
        except Exception:
            skipped += 1
        if progress:
            progress(n, total, name)

    if not vectors:
        raise ValueError("None of the uploaded files could be processed.")

    index = Index(
        folder="Uploaded this session",
        paths=names,
        vectors=np.stack(vectors).astype(np.float32),
        mtimes=np.zeros(len(names), dtype=np.float64),
        thumbs=thumbs,
    )
    return index, {"total": total, "new": len(names), "reused": 0, "skipped": skipped}
