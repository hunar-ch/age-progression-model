"""
Sanity check to run BEFORE building the GUI.

Confirms:
1. The model loads and runs on your device.
2. All 6 alpha steps produce visually distinct, correctly-directed results
   (i.e. -30 actually looks younger, +30 actually looks older).

Usage:
    python test_model.py /path/to/one/face/image.jpg
"""

import sys
import matplotlib.pyplot as plt
from PIL import Image

from model_utils import generate_all_steps, ALPHA_MAP

STEP_ORDER = ["-30", "-20", "-10", "+10", "+20", "+30"]


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_model.py /path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path).convert("RGB")

    print(f"Running all {len(ALPHA_MAP)} steps on {image_path} ...")
    results = generate_all_steps(image)

    fig, axes = plt.subplots(1, len(STEP_ORDER) + 1, figsize=(3 * (len(STEP_ORDER) + 1), 3.5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, label in zip(axes[1:], STEP_ORDER):
        ax.imshow(results[label])
        ax.set_title(f"{label} yrs")
        ax.axis("off")

    plt.tight_layout()
    out_path = "test_output.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison grid to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()