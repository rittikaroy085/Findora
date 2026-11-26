
from PIL import Image

logo = Image.open("pwa/logo.png").convert("RGBA")

for size in (192, 512):
    resized = logo.copy()
    resized.thumbnail((size, size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    x = (size - resized.width) // 2
    y = (size - resized.height) // 2

    canvas.paste(resized, (x, y), resized)
    canvas.save(f"pwa/icons/icon-{size}.png")

print("Findora icons created successfully!")
