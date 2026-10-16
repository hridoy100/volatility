"""Train a custom CNN on an 80/10/10 split of CIFAR-10."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit("Please run this script with Python 3.10 or newer.")

import torch
import torchvision
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def split_indices(targets, seed):
    """Shuffle each class separately so all three sets stay balanced."""
    generator = torch.Generator().manual_seed(seed)
    targets = torch.tensor(targets)
    indices = {"train": [], "validation": [], "test": []}

    for label in range(10):
        class_indices = torch.where(targets == label)[0]
        class_indices = class_indices[
            torch.randperm(len(class_indices), generator=generator)
        ]
        train_end = int(0.8 * len(class_indices))
        validation_end = train_end + int(0.1 * len(class_indices))
        indices["train"].extend(class_indices[:train_end].tolist())
        indices["validation"].extend(
            class_indices[train_end:validation_end].tolist()
        )
        indices["test"].extend(class_indices[validation_end:].tolist())

    return indices


def load_data(data_dir, seed):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    evaluation_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    original_train = datasets.CIFAR10(data_dir, train=True, download=True)
    original_test = datasets.CIFAR10(data_dir, train=False, download=True)
    indices = split_indices(original_train.targets + original_test.targets, seed)
    classes = original_train.classes

    # Both views use the same image order, but only training gets augmentation.
    training_view = ConcatDataset([
        datasets.CIFAR10(data_dir, train=True, transform=train_transform),
        datasets.CIFAR10(data_dir, train=False, transform=train_transform),
    ])
    original_train.transform = evaluation_transform
    original_test.transform = evaluation_transform
    evaluation_view = ConcatDataset([original_train, original_test])

    subsets = {
        "train": Subset(training_view, indices["train"]),
        "validation": Subset(evaluation_view, indices["validation"]),
        "test": Subset(evaluation_view, indices["test"]),
    }
    return subsets, indices, classes


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0:
        parser.error("epochs, batch-size and lr must be positive; workers must be >= 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | {device}")
    subsets, indices, classes = load_data(args.data_dir, args.seed)
    loaders = {
        name: DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            generator=torch.Generator().manual_seed(args.seed),
        )
        for name, subset in subsets.items()
    }
    split_sizes = {name: len(subset) for name, subset in subsets.items()}
    print(f"Split sizes: {split_sizes}")

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_validation_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, loaders["validation"], criterion, device
        )
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.2%} | "
            f"validation loss {validation_loss:.4f}, accuracy {validation_accuracy:.2%}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_loss": validation_loss,
                "classes": classes,
                "seed": args.seed,
                "split_indices": indices,
            }, checkpoint_path)

    # Select the model using validation loss before looking at the test set.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = run_epoch(model, loaders["test"], criterion, device)
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2%}")

    with (args.output_dir / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "seed": args.seed,
        "split_sizes": split_sizes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "best_epoch": checkpoint["epoch"],
        "best_validation_loss": checkpoint["validation_loss"],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved trained model to {checkpoint_path.resolve()}")
    print(f"Saved model and results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
