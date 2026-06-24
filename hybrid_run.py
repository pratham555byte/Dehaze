import os
from pipeline.hybrid_pipeline import HybridPipeline


INPUT_DIR = "input"
OUTPUT_DIR = "results/hybrid"


pipeline = HybridPipeline()


valid_ext = [
    ".jpg",
    ".png",
    ".jpeg",
    ".bmp"
]


for file in os.listdir(INPUT_DIR):
    ext = os.path.splitext(file)[1].lower()

    if ext not in valid_ext:
        continue

    input_path = os.path.join(INPUT_DIR, file)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{os.path.splitext(file)[0]}_hybrid.png"
    )

    pipeline.process_image(
        input_path,
        output_path
    )