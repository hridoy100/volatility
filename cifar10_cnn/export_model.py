"""Export the trained CIFAR-10 checkpoint for TorchScript C++ inference."""

import argparse
from pathlib import Path

import torch

from train import CIFAR10CNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cifar10_torchscript.pt"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("Use a separate output path to preserve the training checkpoint.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CIFAR10CNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scripted_model = torch.jit.script(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(
        scripted_model,
        str(args.output),
        _extra_files={"classes.txt": "\n".join(checkpoint["classes"])},
    )

    # Check the saved archive, including support for different batch sizes.
    exported_model = torch.jit.load(str(args.output), map_location="cpu")
    exported_model.eval()
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for batch_size in (1, 4):
            images = torch.rand(batch_size, 3, 32, 32, generator=generator) * 2 - 1
            torch.testing.assert_close(
                exported_model(images), model(images), rtol=1e-5, atol=1e-6
            )

    print(f"Exported checkpoint from epoch {checkpoint['epoch']}")
    print("Saved TorchScript predictions match the original model (batch sizes 1 and 4).")
    print(f"TorchScript model: {args.output.resolve()}")


if __name__ == "__main__":
    main()
