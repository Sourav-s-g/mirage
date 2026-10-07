"""Writes a small set of random-noise images for testing the pipeline."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

WARNING = (
    "WARNING: this is random-noise dummy data for testing the pipeline only. "
    "It contains no faces. Never report results obtained on it."
)


def fine_noise(rng, size):
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def blocky_noise(rng, size, block=16):
    small = rng.integers(0, 256, (size // block, size // block, 3), dtype=np.uint8)
    return np.kron(small, np.ones((block, block, 1), dtype=np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/dummy")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--train", type=int, default=100, help="images per class")
    parser.add_argument("--valid", type=int, default=30, help="images per class")
    parser.add_argument("--test", type=int, default=70, help="images per class, must exceed the demo holdout")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(WARNING)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    counts = {"train": args.train, "valid": args.valid, "test": args.test}
    for split, n in counts.items():
        for class_name, make in [("real", fine_noise), ("fake", blocky_noise)]:
            folder = out / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                Image.fromarray(make(rng, args.size)).save(folder / f"{i:05d}.jpg", quality=90)
    print(f"wrote {2 * sum(counts.values())} images to {out}")
    print(WARNING)


if __name__ == "__main__":
    main()
