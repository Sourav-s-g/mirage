"""Checks the dataset for metadata cues that separate real from fake without looking at faces."""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = {0: "real", 1: "fake"}


def read_metadata(df, root):
    rows = []
    for rel in df["path"]:
        path = root / rel
        # Image.open only reads the header, so this is fast even for many files
        with Image.open(path) as img:
            rows.append({
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
                "file_size": os.path.getsize(path),
            })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def print_class_report(meta):
    for label, name in CLASS_NAMES.items():
        m = meta[meta["label"] == label]
        print(f"\n{name} ({len(m)} images)")
        dims = (m["width"].astype(str) + "x" + m["height"].astype(str)).value_counts()
        print("dimensions:", dims.head(5).to_dict())
        print("colour mode:", m["mode"].value_counts().to_dict())
        print("format:", m["format"].value_counts().to_dict())
        kb = m["file_size"] / 1024
        q = kb.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(1)
        print("file size KB quantiles:", {f"p{int(k * 100)}": v for k, v in q.items()})


def file_size_auc(meta):
    """Fits logistic regression on log file size (train split) and returns ROC-AUC on val."""
    train = meta[meta["split"] == "train"]
    val = meta[meta["split"] == "val"]
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(np.log(train[["file_size"]]), train["label"])
    scores = model.predict_proba(np.log(val[["file_size"]]))[:, 1]
    return roc_auc_score(val["label"], scores)


def save_sample_grid(df, root, out_path, n=8, seed=42):
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for row, (label, name) in enumerate(CLASS_NAMES.items()):
        sample = df[(df["split"] == "train") & (df["label"] == label)].sample(n=n, random_state=seed)
        for ax, rel in zip(axes[row], sample["path"]):
            ax.imshow(Image.open(root / rel).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row][0].set_ylabel(name, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--figure", default="results/figures/sample_grid.png")
    parser.add_argument("--warn-auc", type=float, default=0.6,
                        help="print a warning if the file-size classifier reaches this val AUC")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    root = manifest.parent
    df = pd.read_csv(manifest)
    meta = read_metadata(df, root)

    print_class_report(meta)

    auc = file_size_auc(meta)
    print(f"\nfile-size-only logistic regression, val ROC-AUC: {auc:.4f}")
    if auc >= args.warn_auc:
        print("WARNING: file size alone separates real from fake. "
              "Models may learn this shortcut instead of facial content.")

    save_sample_grid(df, root, Path(args.figure))
    print(f"sample grid saved to {args.figure}")


if __name__ == "__main__":
    main()
