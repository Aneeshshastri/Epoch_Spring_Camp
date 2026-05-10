import json, pathlib
nb = json.loads(pathlib.Path("Emotion_Recognition.ipynb").read_text(encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
src = nb["cells"][-1]["source"]
print("--- Last cell (first 5 lines) ---")
for line in src[:5]:
    print(repr(line))
print("--- Last cell (last 5 lines) ---")
for line in src[-5:]:
    print(repr(line))
pathlib.Path(__file__).unlink()
