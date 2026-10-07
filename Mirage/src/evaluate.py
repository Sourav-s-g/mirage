"""Evaluation of a trained checkpoint on the test split."""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from src.data.dataset import FaceDataset
from src.data.transforms import build_transform
from src.metrics import compute_metrics, save_metrics
from src.models import get_model


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict(model, loader, device):
    """Returns true labels, probabilities of fake, and forward-pass time per image in ms."""
    model.eval()
    use_amp = device.type == "cuda"
    labels, probs = [], []
    seconds = 0.0
    n_images = 0
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if i == 0:
                model(x)  # warm-up pass so start-up cost is not counted in the timing
            sync(device)
            start = time.perf_counter()
            logits = model(x).view(-1)
            sync(device)
        seconds += time.perf_counter() - start
        n_images += len(x)
        probs.append(torch.sigmoid(logits.float()).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy(), 1000 * seconds / n_images


def save_confusion_matrix(cm, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
    ax.set_xticks([0, 1], ["real", "fake"])
    ax.set_yticks([0, 1], ["real", "fake"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def evaluate_checkpoint(ckpt_path, data_root, out, batch_size, workers, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    name = ckpt["model"]
    img_size = ckpt["img_size"]

    model = get_model(name, img_size, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    manifest = Path(data_root) / "manifest.csv"
    test_ds = FaceDataset(manifest, "test", build_transform(train=False, size=img_size))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    y_true, y_prob, ms_per_image = predict(model, loader, device)

    metrics = compute_metrics(y_true, y_prob)
    folder = Path(out) / name
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    save_metrics(folder / "metrics.json", metrics, name, params, ckpt["train_seconds"],
                 ms_per_image, img_size, ckpt["seed"], ckpt["n_train_images"])
    save_confusion_matrix(metrics["confusion_matrix"], folder / "confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(folder / "roc.csv", index=False)

    print(f"test ({len(test_ds)} images): accuracy {metrics['accuracy']:.4f}, "
          f"roc_auc {metrics['roc_auc']:.4f}, {ms_per_image:.2f} ms/image")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data", help="folder containing manifest.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_checkpoint(args.checkpoint, args.data_root, args.out, args.batch_size,
                        args.workers, device)


if __name__ == "__main__":
    main()
