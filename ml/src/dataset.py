from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PlantDataset(Dataset):
    def __init__(self, csv_path: Path, label_to_idx: dict[str, int], image_size: int = 224):
        self.samples: list[tuple[str, int]] = []
        self.transforms = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(degrees=15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["path"], label_to_idx[row["label"]]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transforms(image), torch.tensor(label, dtype=torch.long)
