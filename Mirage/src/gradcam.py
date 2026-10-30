"""Grad-CAM visualisations for the CNN and ViT checkpoints.

    python -m src.gradcam demo   --checkpoint checkpoints/cnn/best.pt
    python -m src.gradcam mean   --checkpoint checkpoints/cnn/best.pt
    python -m src.gradcam errors --checkpoint checkpoints/cnn/best.pt

Every heatmap is computed for the class the model predicted: it shows the image regions whose
features pushed the output towards that prediction. Grad-CAM scales each heatmap to 0-1 per
image, so brightness compares positions within one image, not confidence between images.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FaceDataset
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transform
from src.models import get_model

CLASS_NAMES = ["real", "fake"]
CAM_NOTE = "Heatmap: Grad-CAM for the predicted class, scaled to 0-1 per image."


class PredictedClassTarget:
    """Grad-CAM target for a one-logit model: the logit for 'fake', or its negative for 'real'."""

    def __init__(self, predicted_label):
        self.sign = 1.0 if predicted_label == 1 else -1.0

    def __call__(self, output):
        return self.sign * output


def load_checkpoint(path, device):
    """Returns the model in eval mode, the model name and the image size from the checkpoint."""
    ckpt = torch.load(path, map_location=device)
    name = ckpt["model"]
    model = get_model(name, ckpt["img_size"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, name, ckpt["img_size"]


def cam_layer_and_reshape(model, name):
    """Target layer and reshape function (None for a CNN) for the given model."""
    if name.startswith("vit"):
        from src.models.vit import cam_target_layer, reshape_transform
        return cam_target_layer(model), reshape_transform
    if name.startswith("cnn"):
        from src.models.cnn import ARCHS
        arch = "resnet34" if name == "cnn_resnet34" else "xception"
        return model.net.get_submodule(ARCHS[arch]["cam_layer"]), None
    raise ValueError(f"Grad-CAM is only set up for the CNN and ViT models, not {name}")


@torch.no_grad()
def predict_probs(model, x):
    """Probability of 'fake' for a batch."""
    return torch.sigmoid(model(x).view(-1).float()).cpu().numpy()


def denormalize(x):
    """(B,3,H,W) normalised tensor -> (B,H,W,3) float image in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()


def compute_cams(model, name, x, predicted):
    """Grad-CAM heatmaps (B,H,W) in [0,1] for a batch, each for its predicted label."""
    layer, reshape = cam_layer_and_reshape(model, name)
    targets = [PredictedClassTarget(int(p)) for p in predicted]
    # float32 and gradients on: Grad-CAM needs a backward pass through the target layer
    with GradCAM(model=model, target_layers=[layer], reshape_transform=reshape) as cam:
        return cam(input_tensor=x.float(), targets=targets)


def heatmap_overlay(model, name, image, device):
    """Heatmap overlay for one normalised image (3,H,W).

    Returns (overlay uint8 (H,W,3), predicted label, probability of fake, heatmap (H,W)).
    """
    x = image.unsqueeze(0).to(device)
    prob = float(predict_probs(model, x)[0])
    label = int(prob >= 0.5)
    cam = compute_cams(model, name, x, [label])[0]
    overlay = show_cam_on_image(denormalize(x)[0].astype(np.float32), cam, use_rgb=True)
    return overlay, label, prob, cam


def sample_indices(ds, label, n, seed):
    """Seeded random indices of up to n images of one class (not hand-picked)."""
    idx = np.flatnonzero(np.array(ds.labels) == label)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())


def batches(ds, indices, batch_size, workers):
    return DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False, num_workers=workers)


def grid_figure(items, title, path, per_row=4):
    """items: list of (image, overlay, caption); each shown as an original/overlay pair in a grid."""
    n_rows = int(np.ceil(len(items) / per_row))
    fig, axes = plt.subplots(n_rows * 2, per_row, figsize=(2.3 * per_row, 2.6 * n_rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (image, overlay, caption) in enumerate(items):
        r, c = divmod(k, per_row)
        axes[2 * r][c].imshow(image)
        axes[2 * r + 1][c].imshow(overlay)
        axes[2 * r + 1][c].set_title(caption, fontsize=7)
    fig.suptitle(f"{title}\n(upper: original, lower: heatmap overlay)  {CAM_NOTE}", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")


def caption_for(label, prob, true_label=None):
    conf = prob if label == 1 else 1 - prob
    text = f"predicted {CLASS_NAMES[label]} ({conf:.2f})"
    if true_label is not None:
        text += f"\ntrue: {CLASS_NAMES[true_label]}"
    return text


def collect_items(model, name, ds, indices, device, batch_size, workers, with_true=False):
    """Heatmap overlays and captions for the given dataset indices."""
    items = []
    for x, y in batches(ds, indices, batch_size, workers):
        x = x.to(device)
        probs = predict_probs(model, x)
        labels = (probs >= 0.5).astype(int)
        cams = compute_cams(model, name, x, labels)
        images = denormalize(x)
        for img, cam, p, lab, true in zip(images, cams, probs, labels, y.tolist()):
            overlay = show_cam_on_image(img.astype(np.float32), cam, use_rgb=True)
            items.append((img, overlay, caption_for(int(lab), float(p), true if with_true else None)))
    return items


def run_demo(args, model, name, img_size, device):
    ds = FaceDataset(Path(args.data_root) / "manifest.csv", "demo",
                     build_transform(False, img_size))
    items = []
    for label in (0, 1):
        idx = sample_indices(ds, label, args.n, args.seed)
        items += collect_items(model, name, ds, idx, device, args.batch_size, args.workers,
                               with_true=True)
    grid_figure(items, f"Grad-CAM, {name}, demo holdout ({args.n} images per class, random draw)",
                Path(args.out) / "figures" / f"gradcam_demo_{name}.png")


def run_mean(args, model, name, img_size, device):
    ds = FaceDataset(Path(args.data_root) / "manifest.csv", "test",
                     build_transform(False, img_size))
    means, mean_imgs, counts = {}, {}, {}
    for label in (0, 1):
        idx = sample_indices(ds, label, args.n, args.seed)
        cam_sum = np.zeros((img_size, img_size), dtype=np.float64)
        img_sum = np.zeros((img_size, img_size, 3), dtype=np.float64)
        for x, _ in batches(ds, idx, args.batch_size, args.workers):
            x = x.to(device)
            labels = (predict_probs(model, x) >= 0.5).astype(int)
            cam_sum += compute_cams(model, name, x, labels).sum(axis=0)
            img_sum += denormalize(x).sum(axis=0)
        counts[label] = len(idx)
        means[label] = cam_sum / len(idx)
        mean_imgs[label] = img_sum / len(idx)

    out = Path(args.out) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"gradcam_mean_{name}.npz", real=means[0], fake=means[1],
             n_real=counts[0], n_fake=counts[1])
    vmax = max(m.max() for m in means.values())
    fig, axes = plt.subplots(2, 2, figsize=(6, 6.4))
    for label in (0, 1):
        axes[0][label].imshow(means[label], cmap="jet", vmin=0, vmax=vmax)
        axes[0][label].set_title(f"mean heatmap, true {CLASS_NAMES[label]} "
                                 f"(n={counts[label]})", fontsize=8)
        axes[1][label].imshow(mean_imgs[label])
        axes[1][label].imshow(means[label], cmap="jet", alpha=0.45, vmin=0, vmax=vmax)
        axes[1][label].set_title("over the mean image of the same images", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Mean Grad-CAM, {name}, test images (random draw)\n"
                 "Average of per-image heatmaps for the predicted class; same colour scale in all panels.",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"gradcam_mean_{name}.png", dpi=130)
    plt.close(fig)
    print(f"saved {out / f'gradcam_mean_{name}.png'} and the arrays in the matching .npz")


def run_errors(args, model, name, img_size, device):
    ds = FaceDataset(Path(args.data_root) / "manifest.csv", "test",
                     build_transform(False, img_size))
    probs = []
    for x, _ in DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers):
        probs.append(predict_probs(model, x.to(device)))
    probs = np.concatenate(probs)
    pred = (probs >= 0.5).astype(int)
    true = np.array(ds.labels)
    wrong = np.flatnonzero(pred != true)
    print(f"{len(wrong)} of {len(ds)} test images misclassified")
    if len(wrong) == 0:
        print("no misclassified images, nothing to draw")
        return
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(wrong, size=min(args.n, len(wrong)), replace=False).tolist())
    items = collect_items(model, name, ds, chosen, device, args.batch_size, args.workers,
                          with_true=True)
    grid_figure(items, f"Misclassified test images, {name} ({len(items)} of {len(wrong)}, random draw)",
                Path(args.out) / "figures" / f"gradcam_errors_{name}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["demo", "mean", "errors"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data", help="folder containing manifest.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--n", type=int, default=None,
                        help="images per class (demo: 8, mean: 500); total misclassified images "
                             "drawn for errors (default 16)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n is None:
        args.n = {"demo": 8, "mean": 500, "errors": 16}[args.command]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, name, img_size = load_checkpoint(args.checkpoint, device)
    {"demo": run_demo, "mean": run_mean, "errors": run_errors}[args.command](
        args, model, name, img_size, device)


if __name__ == "__main__":
    main()
