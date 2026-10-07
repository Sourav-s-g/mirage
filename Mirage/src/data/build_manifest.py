"""Builds a manifest CSV (path, label, split) from a train/valid/test folder layout."""
import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd

SPLIT_NAMES = {"train": "train", "valid": "val", "test": "test"}
LABELS = {"real": 0, "fake": 1}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(root):
    rows = []
    for folder, split in SPLIT_NAMES.items():
        for class_name, label in LABELS.items():
            class_dir = root / folder / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"missing folder: {class_dir}")
            # sorted so the demo sample below is the same on every machine
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in IMAGE_EXTS:
                    rows.append({"path": p, "label": label, "split": split})
    return pd.DataFrame(rows)


def take_demo_holdout(df, per_class, seed):
    test = df[df["split"] == "test"]
    for label in LABELS.values():
        picked = test[test["label"] == label].sample(n=per_class, random_state=seed)
        df.loc[picked.index, "split"] = "demo"
    return df


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_no_overlap(df):
    """Fails if a path, or identical file content, appears in more than one split."""
    if df["path"].duplicated().any():
        raise ValueError("the same path appears more than once in the manifest")

    hashes = df["path"].map(file_md5)
    splits_per_hash = df.groupby(hashes)["split"].nunique()
    shared = splits_per_hash[splits_per_hash > 1]
    if len(shared) > 0:
        raise ValueError(f"{len(shared)} identical files appear in more than one split")

    dup_within = hashes.duplicated().sum()
    if dup_within:
        print(f"note: {dup_within} files are exact duplicates of another file in the same split")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="folder containing train/, valid/ and test/")
    parser.add_argument("--out", default="data/manifest.csv")
    parser.add_argument("--demo-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    df = list_images(root)
    df = take_demo_holdout(df, args.demo_per_class, args.seed)
    check_no_overlap(df)

    # paths are stored relative to the manifest so the data folder can be moved as a whole
    df["path"] = df["path"].map(lambda p: Path(os.path.relpath(p, out.parent)).as_posix())
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    counts.columns = ["real" if c == 0 else "fake" for c in counts.columns]
    print(counts.reindex(["train", "val", "test", "demo"]))
    print(f"total {len(df)}, written to {out}")


if __name__ == "__main__":
    main()
