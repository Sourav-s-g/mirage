"""Dataset of real and fake face images read from the manifest."""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, manifest, split, transform=None, limit=None, seed=42):
        """If limit is set, use a class-balanced random subset of about that many images."""
        df = pd.read_csv(manifest)
        df = df[df["split"] == split]
        if limit is not None:
            per_class = limit // 2
            parts = []
            for label in sorted(df["label"].unique()):
                g = df[df["label"] == label]
                parts.append(g.sample(n=min(per_class, len(g)), random_state=seed))
            df = pd.concat(parts)
        self.root = Path(manifest).parent
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.root / self.paths[i]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[i]
