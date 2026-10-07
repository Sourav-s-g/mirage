"""Trains the MLP baseline on cached image features and tests the best checkpoint."""
import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

from src.data.dataset import FaceDataset
from src.evaluate import save_confusion_matrix, sync
from src.metrics import compute_metrics, save_metrics
from src.models import mlp
from src.train import CKPT_DIR, set_seed


def get_features(ds, split, cache_dir):
    """Features of every image in ds for both variants, computed once and cached as .npy."""
    # the key depends on the exact list of images, so a different --limit gets its own cache
    key = hashlib.md5("\n".join(ds.paths).encode()).hexdigest()[:8]
    files = {v: cache_dir / f"{split}_{key}_{v}.npy" for v in mlp.FEATURES}
    if all(f.exists() for f in files.values()):
        return {v: np.load(f) for v, f in files.items()}

    rows = {v: [] for v in mlp.FEATURES}
    for i in tqdm(range(len(ds)), desc=f"features {split}"):
        img, _ = ds[i]
        for v, fn in mlp.FEATURES.items():
            rows[v].append(fn(img))
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for v in mlp.FEATURES:
        out[v] = np.stack(rows[v]).astype(np.float32)
        np.save(files[v], out[v])
    return out


def predict_probs(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x).view(-1)).cpu().numpy()


def ms_per_image(ds, features, mean, scale, model, device, n=200):
    """Time per image of feature extraction, scaling and the MLP forward pass (image decoding excluded)."""
    imgs = [ds[i][0] for i in range(min(n, len(ds)))]
    fn = mlp.FEATURES[features]
    model.eval()

    def run(img):
        x = torch.from_numpy(((fn(img) - mean) / scale).astype(np.float32))
        with torch.no_grad():
            model(x.to(device)[None])

    run(imgs[0])  # warm-up
    sync(device)
    start = time.perf_counter()
    for img in imgs:
        run(img)
    sync(device)
    return 1000 * (time.perf_counter() - start) / len(imgs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, choices=list(mlp.FEATURES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None,
                        help="train on a class-balanced random subset of this many images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data", help="folder containing manifest.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"mlp_{args.features}"

    manifest = Path(args.data_root) / "manifest.csv"
    cache_dir = Path(args.data_root) / "features"
    datasets = {
        "train": FaceDataset(manifest, "train", limit=args.limit, seed=args.seed),
        "val": FaceDataset(manifest, "val"),
        "test": FaceDataset(manifest, "test"),
    }
    raw = {s: get_features(ds, s, cache_dir)[args.features] for s, ds in datasets.items()}

    # the scaler sees the training features only, so val and test statistics do not leak in
    scaler = StandardScaler().fit(raw["train"])
    x = {s: torch.from_numpy(scaler.transform(f).astype(np.float32)).to(device)
         for s, f in raw.items()}
    y = {s: torch.tensor(ds.labels, dtype=torch.float32, device=device)
         for s, ds in datasets.items()}
    n_train = len(datasets["train"])

    model = mlp.build(x["train"].shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}: {x['train'].shape[1]} features, {n_params} trainable parameters, "
          f"{n_train} train images, {len(datasets['val'])} val images, device {device}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(args.seed)

    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / name / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    start = time.time()
    with open(out_dir / "log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_accuracy", "val_roc_auc", "elapsed_seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for idx in torch.randperm(n_train, generator=generator).split(args.batch_size):
                optimizer.zero_grad()
                loss = criterion(model(x["train"][idx]).view(-1), y["train"][idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
            scheduler.step()

            y_true = y["val"].cpu().numpy()
            y_prob = predict_probs(model, x["val"])
            val_acc = float(((y_prob >= 0.5) == y_true).mean())
            val_auc = float(roc_auc_score(y_true, y_prob))
            log.writerow([epoch, total_loss / n_train, val_acc, val_auc, time.time() - start])
            f.flush()
            print(f"epoch {epoch}/{args.epochs}: train loss {total_loss / n_train:.4f}, "
                  f"val accuracy {val_acc:.4f}, val roc_auc {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    "model": name,
                    "model_state": model.state_dict(),
                    "in_dim": x["train"].shape[1],
                    "scaler_mean": torch.from_numpy(scaler.mean_),
                    "scaler_scale": torch.from_numpy(scaler.scale_),
                    "seed": args.seed,
                    "n_train_images": n_train,
                    "train_seconds": time.time() - start,
                    "epoch": epoch,
                }, ckpt_path)

    train_seconds = time.time() - start
    # the checkpoint was written mid-run, so store the total training time now
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt["train_seconds"] = train_seconds
    torch.save(ckpt, ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    print(f"best val roc_auc {best_auc:.4f} at epoch {ckpt['epoch']}, evaluating on test")

    y_true = y["test"].cpu().numpy()
    y_prob = predict_probs(model, x["test"])
    metrics = compute_metrics(y_true, y_prob)
    ms = ms_per_image(datasets["test"], args.features, scaler.mean_, scaler.scale_, model, device)
    # the thumbnail is 32 px; the spectrum and histograms use the image at its original size
    img_size = mlp.THUMB_SIZE if args.features == "pixels" else datasets["train"][0][0].width
    save_metrics(out_dir / "metrics.json", metrics, name, n_params, train_seconds, ms,
                 img_size, args.seed, n_train)
    save_confusion_matrix(metrics["confusion_matrix"], out_dir / "confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(out_dir / "roc.csv", index=False)
    print(f"test ({len(y_true)} images): accuracy {metrics['accuracy']:.4f}, "
          f"roc_auc {metrics['roc_auc']:.4f}, {ms:.2f} ms/image")


if __name__ == "__main__":
    main()
