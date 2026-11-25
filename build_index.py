from pathlib import Path
import numpy as np
import onnxruntime as ort

from transformers import CLIPTokenizer, CLIPImageProcessor
from PIL import Image

BASE = Path(__file__).resolve().parent

VISION_DIR = BASE / "models" / "clip" / "vision"

# Load vision model
vision = ort.InferenceSession(
    str(VISION_DIR / "model.onnx"),
    providers=["CPUExecutionProvider"]
)

processor = CLIPImageProcessor.from_pretrained(
    str(VISION_DIR), local_files_only=True
)

# Select folder
folder_path = input("Enter image folder path: ").strip().strip('"')
folder = Path(folder_path)

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

images = [
    p for p in folder.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

print(f"\nFound {len(images)} images.")

vectors = []
paths = []

for path in images:
    try:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="np")

        vector = vision.run(
            ["image_embeds"],
            {"pixel_values": inputs["pixel_values"]}
        )[0][0]

        vector = vector / np.linalg.norm(vector)

        vectors.append(vector)
        paths.append(str(path))

        print("Indexed:", path.name)

    except Exception as e:
        print("Skipped:", path.name, "|", e)

if vectors:
    np.savez(
        BASE / "image_index.npz",
        vectors=np.array(vectors, dtype=np.float32),
        paths=np.array(paths)
    )

    print(f"\nSaved index for {len(vectors)} images!")
    print("File: image_index.npz")
else:
    print("No valid images indexed.")