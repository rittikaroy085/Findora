from pathlib import Path
import numpy as np
import onnxruntime as ort

from transformers import CLIPTokenizer, CLIPImageProcessor
from PIL import Image

BASE = Path(__file__).resolve().parent

VISION_DIR = BASE / "models" / "clip" / "vision"
TEXT_DIR = BASE / "models" / "clip" / "text"

# Load models
vision = ort.InferenceSession(
    str(VISION_DIR / "model.onnx"),
    providers=["CPUExecutionProvider"]
)

text_model = ort.InferenceSession(
    str(TEXT_DIR / "model.onnx"),
    providers=["CPUExecutionProvider"]
)

# Load preprocessing tools
tokenizer = CLIPTokenizer.from_pretrained(
    str(TEXT_DIR), local_files_only=True
)

processor = CLIPImageProcessor.from_pretrained(
    str(VISION_DIR), local_files_only=True
)

# Select image folder
folder_path = input("Enter image folder path: ").strip().strip('"')
folder = Path(folder_path)

# Find supported images
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

images = [
    p for p in folder.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

if not images:
    print("No images found.")
    exit()

print(f"\nFound {len(images)} images.")

# Generate image embeddings
image_vectors = []
valid_images = []

for path in images:
    try:
        image = Image.open(path).convert("RGB")

        inputs = processor(images=image, return_tensors="np")

        vector = vision.run(
            ["image_embeds"],
            {"pixel_values": inputs["pixel_values"]}
        )[0][0]

        vector = vector / np.linalg.norm(vector)

        image_vectors.append(vector)
        valid_images.append(path)

        print("Indexed:", path.name)

    except Exception as e:
        print("Skipped:", path.name, "|", e)

if not image_vectors:
    print("Could not process any images.")
    exit()

# Search query
query = input("\nWhat do you want to find? ")

tokens = tokenizer(
    query,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="np"
)

text_vector = text_model.run(
    ["text_embeds"],
    {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    }
)[0][0]

text_vector = text_vector / np.linalg.norm(text_vector)

# Rank images
scores = np.dot(
    np.array(image_vectors),
    text_vector
)

ranked = sorted(
    zip(valid_images, scores),
    key=lambda x: x[1],
    reverse=True
)

# Display results
print("\n========== FINDORA RESULTS ==========")

for rank, (path, score) in enumerate(ranked, 1):
    print(f"{rank}. {path.name} | Score: {score:.4f}")
    print(f"   {path}")

print("====================================")