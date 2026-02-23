"""Create a unified dataset index from PlantNet/iNaturalist/LeafSnap style folders."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_index(dataset_root: Path, out_csv: Path) -> None:
    rows: list[tuple[str, str]] = []
    for class_dir in dataset_root.iterdir():
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for img in class_dir.glob("**/*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                rows.append((str(img.resolve()), label))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/index.csv"))
    args = parser.parse_args()
    build_index(args.dataset_root, args.out)


if __name__ == "__main__":
    main()
