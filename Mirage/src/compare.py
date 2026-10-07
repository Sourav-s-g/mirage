"""Compares all trained models: a markdown table and an overlaid ROC plot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_results(results_dir):
    runs = []
    for path in sorted(Path(results_dir).glob("*/metrics.json")):
        with open(path) as f:
            runs.append((path.parent, json.load(f)))
    return runs


def print_table(runs):
    header = ["model", "accuracy", "precision", "recall", "F1", "ROC-AUC",
              "params", "ms/image", "image size", "training images used"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for _, m in runs:
        row = [m["model"], f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
               f"{m['recall']:.4f}", f"{m['f1']:.4f}", f"{m['roc_auc']:.4f}",
               str(m["trainable_params"]), f"{m['ms_per_image']:.2f}", str(m["img_size"]),
               str(m["n_train_images"])]
        print("| " + " | ".join(row) + " |")


def save_roc_plot(runs, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for folder, m in runs:
        roc_file = folder / "roc.csv"
        if not roc_file.exists():
            continue
        roc = pd.read_csv(roc_file)
        ax.plot(roc["fpr"], roc["tpr"], label=f"{m['model']} (AUC {m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    runs = load_results(args.results)
    if not runs:
        print(f"no metrics.json found under {args.results}")
        return
    print_table(runs)
    out = Path(args.results) / "figures" / "roc_comparison.png"
    save_roc_plot(runs, out)
    print(f"\nROC plot saved to {out}")


if __name__ == "__main__":
    main()
