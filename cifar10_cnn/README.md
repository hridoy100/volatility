# CIFAR-10 with a custom CNN

This is a small image classifier built directly with `torch.nn`. It uses no
prebuilt networks or pretrained weights. The code follows the basic workflow in
the [PyTorch 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).

Use Python **3.10 or newer**; Python 3.12 is a suitable choice. The dependencies
are pinned to [PyTorch 2.14.1](https://github.com/pytorch/pytorch/releases/tag/v2.14.1)
and [torchvision 0.29.1](https://github.com/pytorch/vision/releases/tag/v0.29.1),
the latest stable releases checked on October 16, 2026.

## Run

From this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

The script downloads CIFAR-10 on the first run and automatically uses CUDA,
Apple's MPS backend, or the CPU. For an NVIDIA GPU, use the appropriate wheel
index from the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

Training defaults to 25 epochs, batch size 128, and learning rate 0.001:

```bash
python train.py --epochs 30 --batch-size 128 --lr 0.001
```

To check that the complete pipeline runs, use `python train.py --epochs 1`.
This is only a quick check; train for the full run before reporting final results.
Use `--workers 0` if your environment has trouble with multiprocessing.

## Training on Argon

Create the Python environment on Argon using the setup steps above; copy the
source files rather than the Mac's `.venv` folder. After activating that
environment, run this command from this folder in your training job:

```bash
python -u train.py --epochs 25 --output-dir outputs
```

The trained checkpoint is saved as **`outputs/best_model.pt`**. It is updated
whenever validation loss improves, so a completed best checkpoint is available
before the entire run finishes. It contains the weights and metadata for the
best validation epoch, which is also the model used for final testing.

To reload that checkpoint later, keep `train.py` with it and use:

```python
import torch
from train import CIFAR10CNN

checkpoint = torch.load("outputs/best_model.pt", map_location="cpu", weights_only=True)
model = CIFAR10CNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Using `map_location="cpu"` lets you load weights saved on Argon's GPU onto a
machine without a GPU.

## Data split

CIFAR-10 has 60,000 images. Its original 50,000-image training set and
10,000-image test set are pooled and repartitioned to meet the requested ratio:

| Set | Images | Images per class | Share |
| --- | ---: | ---: | ---: |
| Training | 48,000 | 4,800 | 80% |
| Validation | 6,000 | 600 | 10% |
| Testing | 6,000 | 600 | 10% |

The split is stratified, disjoint, and reproducible with seed 42. This is a
custom split, so its scores are not directly comparable with results using the
official 10,000-image test set. Only training images receive random crops and
horizontal flips. Every image is converted to a tensor and normalized to
[-1, 1] using fixed constants, with no statistics fitted on held-out images.

## Network and training

Three convolutional blocks learn 32, 64, and 128 feature channels. Each block
contains two 3x3 convolutions with ReLU activations, batch normalization after
the first convolution, and 2x2 max pooling. Pooling reduces the spatial size
from 32x32 to 16x16, 8x8, and finally 4x4. A 256-unit dense layer and dropout
feed the final 10-class output.

The network returns raw logits because `CrossEntropyLoss` handles the
log-softmax calculation. AdamW updates the weights, and a cosine schedule
reduces the learning rate over the run. Validation runs after each epoch in
evaluation mode with gradients disabled. The checkpoint with the lowest
validation loss is reloaded for one final test evaluation.

The script saves three files in `outputs/`:

- `best_model.pt`: model weights, class names, selected epoch, seed, and split indices.
- `history.csv`: training and validation loss and accuracy for each epoch.
- `metrics.json`: final test metrics, split sizes, settings, and package versions.

Accuracy values in the saved files are fractions between 0 and 1. Console output
shows percentages. The seed fixes the split and random initialization; small
numerical differences can still occur across devices and PyTorch backends.
