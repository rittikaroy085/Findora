from pathlib import Path
import onnxruntime as ort

BASE = Path(__file__).resolve().parent

models = {
    "VISION ENCODER": BASE / "models" / "clip" / "vision" / "model.onnx",
    "TEXT ENCODER": BASE / "models" / "clip" / "text" / "model.onnx"
}

for name, path in models.items():
    print(f"\nChecking {name}...")

    if not path.exists():
        print(f"ERROR: Model not found at {path}")
        continue

    try:
        session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"]
        )

        print("STATUS: LOADED SUCCESSFULLY")

        print("Inputs:")
        for item in session.get_inputs():
            print(f"  {item.name}: {item.shape}")

        print("Outputs:")
        for item in session.get_outputs():
            print(f"  {item.name}: {item.shape}")

        del session

    except Exception as error:
        print("ERROR:", error)

print("\nMODEL CHECK COMPLETE")
