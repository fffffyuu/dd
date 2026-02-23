"""Export PyTorch checkpoints to ONNX and TFLite (via ONNX-TF bridge optional)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.models as models


def build_model(num_classes: int):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--onnx-out", type=Path, default=Path("artifacts/model.onnx"))
    args = parser.parse_args()

    model = build_model(args.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    args.onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.onnx_out,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX exported to {args.onnx_out}")
    print("For TFLite/CoreML, convert ONNX using onnx-tf/coremltools pipelines.")


if __name__ == "__main__":
    main()
