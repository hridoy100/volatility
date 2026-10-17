"""Write the checkpoint's held-out CIFAR-10 images for C++ inference."""

import argparse
import json
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/cifar10_test.bin"))
    args = parser.parse_args()
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("The output must not overwrite the checkpoint.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    split = checkpoint["split_indices"]
    indices = split["test"]
    if not indices or len(set(indices)) != len(indices):
        parser.error("The checkpoint must contain distinct test indices.")
    if set(indices) & (set(split["train"]) | set(split["validation"])):
        parser.error("The test split overlaps the training or validation split.")

    original_train = CIFAR10(args.data_dir, train=True, download=False)
    original_test = CIFAR10(args.data_dir, train=False, download=False)
    if min(indices) < 0 or max(indices) >= len(original_train) + len(original_test):
        parser.error("The checkpoint contains an invalid CIFAR-10 index.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        for index in indices:
            if index < len(original_train):
                dataset, offset = original_train, index
            else:
                dataset, offset = original_test, index - len(original_train)
            file.write(bytes([dataset.targets[offset]]))
            file.write(dataset.data[offset].transpose(2, 0, 1).tobytes())

    manifest_path = args.output.with_suffix(".indices.json")
    manifest_path.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "dataset_order": "original CIFAR-10 training set, then original test set",
        "indices": indices,
    }, indent=2) + "\n")
    print(f"Saved {len(indices)} held-out test images to {args.output.resolve()}")
    print(f"Original dataset indices: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
