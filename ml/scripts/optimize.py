"""Apply dynamic quantization and structured pruning for mobile inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
import torchvision.models as models


def build_model(num_classes: int):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/model_optimized.pt"))
    args = parser.parse_args()

    model = build_model(args.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=0.2, n=2, dim=0)
            prune.remove(module, "weight")

    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized.state_dict(), args.out)
    print(f"Optimized model checkpoint: {args.out}")


if __name__ == "__main__":
    main()
