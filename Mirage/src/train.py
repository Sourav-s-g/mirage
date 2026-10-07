"""Trains one model on the manifest splits, keeps the best checkpoint by val ROC-AUC, then tests it."""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import FaceDataset
from src.data.transforms import build_transform
from src.evaluate import evaluate_checkpoint, predict
from src.models import get_model

CKPT_DIR = Path("checkpoints")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp,
                    clip_norm):
    model.train()
    total_loss = 0.0
    n_images = 0
    for x, y in loader:
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x).view(-1)
        loss = criterion(logits.float(), y)
        scaler.scale(loss).backward()
        if clip_norm is not None:
            scaler.unscale_(optimizer)  # clip the real gradients, not the loss-scaled ones
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * len(x)
        n_images += len(x)
    return total_loss / n_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "cnn_resnet34", "lstm", "vit", "vit_base"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=None,
                        help="train on a class-balanced random subset of this many images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data", help="folder containing manifest.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    manifest = Path(args.data_root) / "manifest.csv"
    train_ds = FaceDataset(manifest, "train", build_transform(True, args.img_size),
                           limit=args.limit, seed=args.seed)
    val_ds = FaceDataset(manifest, "val", build_transform(False, args.img_size))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    model = get_model(args.model, args.img_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.model}: {n_params} trainable parameters, {len(train_ds)} train images, "
          f"{len(val_ds)} val images, device {device}")

    criterion = nn.BCEWithLogitsLoss()
    # only models that ask for it (the LSTM) get gradient clipping
    clip_norm = getattr(model, "grad_clip_norm", None)
    # a model can give separate learning rates to its backbone and head
    params = model.param_groups(args.lr) if hasattr(model, "param_groups") else model.parameters()
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    # cosine decay, stepped every batch; a model can ask for a linear warmup at the start
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(getattr(model, "warmup_fraction", 0.0) * total_steps)

    def lr_scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / args.model / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    start = time.time()
    with open(out_dir / "log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_accuracy", "val_roc_auc", "elapsed_seconds"])
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                         scaler, device, use_amp, clip_norm)
            y_true, y_prob, _ = predict(model, val_loader, device)
            val_acc = float(((y_prob >= 0.5) == y_true).mean())
            val_auc = float(roc_auc_score(y_true, y_prob))
            log.writerow([epoch, train_loss, val_acc, val_auc, time.time() - start])
            f.flush()
            print(f"epoch {epoch}/{args.epochs}: train loss {train_loss:.4f}, "
                  f"val accuracy {val_acc:.4f}, val roc_auc {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    "model": args.model,
                    "model_state": model.state_dict(),
                    "img_size": args.img_size,
                    "seed": args.seed,
                    "n_train_images": len(train_ds),
                    "train_seconds": time.time() - start,
                    "epoch": epoch,
                }, ckpt_path)

    # the checkpoint was written mid-run, so store the total training time now
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt["train_seconds"] = time.time() - start
    torch.save(ckpt, ckpt_path)

    print(f"best val roc_auc {best_auc:.4f} at epoch {ckpt['epoch']}, evaluating on test")
    evaluate_checkpoint(ckpt_path, args.data_root, args.out, args.batch_size,
                        args.workers, device)


if __name__ == "__main__":
    main()
