
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

# Get image path
image_path = input("Enter image path: ").strip().strip('"')

image = Image.open(image_path).convert("RGB")

# Generate image embedding
image_inputs = processor(
    images=image, return_tensors="np"
)

image_vector = vision.run(
    ["image_embeds"],
    {"pixel_values": image_inputs["pixel_values"]}
)[0][0]

# Get search description
query = input("Describe what you want to find: ")

tokens = tokenizer(
    query,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="np"
)

# Generate text embedding
text_vector = text_model.run(
    ["text_embeds"],
    {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    }
)[0][0]

# Normalize vectors
image_vector /= np.linalg.norm(image_vector)
text_vector /= np.linalg.norm(text_vector)

# Calculate similarity
score = float(np.dot(image_vector, text_vector))

print("\n========== FINDORA SEARCH ==========")
print("Image:", image_path)
print("Query:", query)
print("Similarity score:", round(score, 4))
print("===================================")
