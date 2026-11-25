
from pathlib import Path
from transformers import CLIPTokenizer, CLIPImageProcessor
from PIL import Image

BASE = Path(__file__).resolve().parent

VISION_DIR = BASE / "models" / "clip" / "vision"
TEXT_DIR = BASE / "models" / "clip" / "text"

print("Loading text tokenizer...")

tokenizer = CLIPTokenizer.from_pretrained(
    str(TEXT_DIR),
    local_files_only=True
)

print("TEXT TOKENIZER: READY")

print("\nLoading image processor...")

processor = CLIPImageProcessor.from_pretrained(
    str(VISION_DIR),
    local_files_only=True
)

print("IMAGE PROCESSOR: READY")

# Test text preprocessing
text = "A dog playing in a green park"

text_inputs = tokenizer(
    text,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="np"
)

print("\nText input shape:", text_inputs["input_ids"].shape)

# Test image preprocessing with a sample image
sample_image = Image.new("RGB", (300, 300), color="blue")

image_inputs = processor(
    images=sample_image,
    return_tensors="np"
)

print("Image input shape:", image_inputs["pixel_values"].shape)

print("\nPREPROCESSING TEST COMPLETE")
