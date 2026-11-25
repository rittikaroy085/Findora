
from pathlib import Path
import numpy as np
import onnxruntime as ort

from transformers import CLIPTokenizer, CLIPImageProcessor
from PIL import Image

BASE = Path(__file__).resolve().parent

VISION_DIR = BASE / "models" / "clip" / "vision"
TEXT_DIR = BASE / "models" / "clip" / "text"

# Load preprocessing components
tokenizer = CLIPTokenizer.from_pretrained(
    str(TEXT_DIR), local_files_only=True
)

processor = CLIPImageProcessor.from_pretrained(
    str(VISION_DIR), local_files_only=True
)

# Load ONNX models
vision_session = ort.InferenceSession(
    str(VISION_DIR / "model.onnx"),
    providers=["CPUExecutionProvider"]
)

text_session = ort.InferenceSession(
    str(TEXT_DIR / "model.onnx"),
    providers=["CPUExecutionProvider"]
)

# Prepare a sample image
image = Image.new("RGB", (300, 300), color="blue")

image_inputs = processor(
    images=image, return_tensors="np"
)

# Generate image embedding
image_output = vision_session.run(
    ["image_embeds"],
    {"pixel_values": image_inputs["pixel_values"]}
)[0]

# Prepare a sample text
text = "A blue square"

text_inputs = tokenizer(
    text,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="np"
)

# Generate text embedding
text_output = text_session.run(
    ["text_embeds"],
    {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"]
    }
)[0]

# Normalize embeddings
image_vector = image_output[0]
text_vector = text_output[0]

image_vector = image_vector / np.linalg.norm(image_vector)
text_vector = text_vector / np.linalg.norm(text_vector)

# Calculate cosine similarity
similarity = np.dot(image_vector, text_vector)

print("\nIMAGE EMBEDDING SHAPE:", image_output.shape)
print("TEXT EMBEDDING SHAPE:", text_output.shape)
print("COSINE SIMILARITY:", float(similarity))

print("\nEMBEDDING TEST COMPLETE!")
