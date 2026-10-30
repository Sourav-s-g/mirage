"""Precomputes the demo outputs for every demo holdout image as static files, for use if the live app fails.

    python demo/make_backup.py --checkpoint checkpoints/cnn/best.pt

Writes to results/demo_backup/: one panel per image (original, heatmap overlay, true label, prediction
and confidence) and summary.csv. These panels show the true labels, so only open them to reveal.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from demo.app import CAM_NOTE, Classifier, load_demo_rows
from src.gradcam import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/demo_backup")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    clf = Classifier(args.checkpoint, args.device)
    rows = load_demo_rows(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for i, (path, true) in enumerate(rows):
        image = Image.open(path).convert("RGB")
        r = clf.analyse(image)
        name = f"demo_{i:03d}"
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.4))
        axes[0].imshow(image)
        axes[0].set_title("original", fontsize=9)
        axes[1].imshow(r["overlay"])
        axes[1].set_title("heatmap overlay", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"true: {CLASS_NAMES[true]}   |   model: {r['text']}", fontsize=9)
        fig.text(0.5, 0.01, CAM_NOTE, ha="center", fontsize=6, wrap=True)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out / f"{name}.png", dpi=130)
        plt.close(fig)
        records.append({"id": name, "source": str(path.relative_to(Path(args.data_root))),
                        "true_label": CLASS_NAMES[true], "predicted": CLASS_NAMES[r["label"]],
                        "confidence": round(r["confidence"], 4), "model": clf.name})

    df = pd.DataFrame(records)
    df.to_csv(out / "summary.csv", index=False)
    print(f"saved {len(df)} panels and summary.csv to {out}; "
          f"accuracy on these images {(df['true_label'] == df['predicted']).mean():.3f}")


if __name__ == "__main__":
    main()
