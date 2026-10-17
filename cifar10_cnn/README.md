# CIFAR-10 with a custom CNN

This is a small image classifier built directly with `torch.nn`. It uses no
prebuilt networks or pretrained weights. The code follows the basic workflow in
the [PyTorch 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).

Use Python **3.10 or newer**; Python 3.12 is a suitable choice. The dependencies
are pinned to [PyTorch 2.14.1](https://github.com/pytorch/pytorch/releases/tag/v2.14.1)
and [torchvision 0.29.1](https://github.com/pytorch/vision/releases/tag/v0.29.1),
the latest stable releases checked on October 16, 2026.

## Assignment workflow

1. **Build and train a custom CNN.** Define the network directly with
   `torch.nn` in [train.py](train.py), using no prebuilt architectures or
   pretrained weights. Follow the
   [PyTorch 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
   for the training workflow. Use CIFAR-10 with an 80/10/10 split: 48,000
   training images, 6,000 validation images, and 6,000 test images. Use the
   most recent stable PyTorch release at setup time and record its version;
   the versions used for this project are pinned above. Python must be newer
   than 3.9; this implementation requires Python 3.10 or newer. Save the best
   trained checkpoint as `outputs/best_model.pt`.

2. **Export the trained model for C++.** Follow the TorchScript workflow in
   the [C++ export tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_export.html).
   Run [export_model.py](export_model.py) to load the trained checkpoint,
   script the model, and save `outputs/cifar10_torchscript.pt`. This archive
   contains the model code and weights needed by `torch::jit::load`.

3. **Build a C++ inference executable with full debugging information.**
   Use [cpp/main.cpp](cpp/main.cpp) to load the TorchScript archive and run
   inference on the held-out test images prepared by
   [prepare_test_images.py](prepare_test_images.py). Configure a Debug build
   using [cpp/CMakeLists.txt](cpp/CMakeLists.txt), which explicitly enables
   full debug information and disables optimization. On Argon, the completed
   Apptainer build produced `cpp/build-apptainer/cifar10_inference`.

4. **Inspect inference using GDB.** Start GDB with the executable, set the
   model and test-image arguments, and break on the model's forward call:
   `auto logits = model.forward({images}).toTensor();`. Run the program and
   use `next` to execute that call. In the completed Argon session, GDB
   stopped at line 91 and advanced to line 92 after the forward pass. Follow
   the [Apptainer and GDB guide](ARGON.md) for the commands and recorded output.
   Then [inspect the PID, object addresses, layer descriptions, and weight
   storage](GDB_INSPECTION.md), using the PyTorch GitHub commit matching the
   container's actual library version.

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

The C++ build and GDB inspection were completed inside Apptainer. See
[Argon: Apptainer build and GDB](ARGON.md) for that workflow and the runtime
library issue encountered during native setup. The guide records the exact
`pytorch-cuda12.6.sif` path, its NVIDIA `24.10-py3` source, and the launch
command. That base image ships an older PyTorch version than the training
environment recorded here; check the container's actual version when
reproducing the build.

Create the Python environment on Argon using the setup steps above; copy the
source files rather than the Mac's `.venv` folder. After activating that
environment, run this command from this folder in your training job:

```bash
python -u train.py --epochs 25 --output-dir outputs
```

If the log reports `PyTorch 2.14.1+cu130 | cpu` and says the NVIDIA driver is
too old, stop the CPU run. In the same Python environment, install the CUDA
12.6 builds of the pinned packages:

```bash
python -m pip install --upgrade torch==2.14.1+cu126 torchvision==0.29.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

These builds are listed in the official [PyTorch](https://download.pytorch.org/whl/cu126/torch/)
and [torchvision](https://download.pytorch.org/whl/cu126/torchvision/) wheel indexes.
[NVIDIA's compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
allows CUDA 12.x minor-version compatibility on supported drivers starting at
525, with feature limitations. CUDA 13.x requires driver 580 or newer. GPU model
support and the actual driver still need to be checked on your assigned node.

Inside your allocated GPU job, verify the environment and a GPU convolution:

```bash
nvidia-smi
python - <<'PY'
import torch

print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is unavailable in this environment"
print("GPU:", torch.cuda.get_device_name(0))
layer = torch.nn.Conv2d(3, 8, kernel_size=3).cuda()
images = torch.randn(2, 3, 32, 32, device="cuda")
layer(images).sum().backward()
torch.cuda.synchronize()
print("GPU forward and backward pass succeeded")
PY
```

After this succeeds, rerun the training command. Its startup line should report
`cuda`. The script already moves the model and each batch to that device.

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

The recorded 25-epoch Argon run used Python 3.10.8 and PyTorch 2.14.1+cu126.
Its held-out test accuracy was **87.08%** (5,225 of 6,000 images), as recorded
in [outputs/metrics.json](outputs/metrics.json).

## Export to TorchScript and load from C++

After training, export the saved checkpoint from this folder:

```bash
python export_model.py --checkpoint outputs/best_model.pt --output outputs/cifar10_torchscript.pt
```

`best_model.pt` is a Python training checkpoint containing a state dictionary.
`cifar10_torchscript.pt` contains the compiled model code, trained weights, and
class names. Load the latter with `torch::jit::load` in C++; it does not need
Python or `train.py` at inference time.

The exporter loads GPU-trained weights onto the CPU, calls `eval()` to disable
dropout and use batch normalization's saved statistics, scripts the custom
network with `torch.jit.script`, and saves it with `torch.jit.save`. It then
reloads the archive and checks its outputs against the original model for
batch sizes 1 and 4. It does not retrain the network or download CIFAR-10.

This follows the TorchScript serialization and C++ loading workflow from the
[C++ export tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_export.html).
That page now marks TorchScript deprecated; this project uses it to meet the
assignment's explicit TorchScript requirement.

### Build the C++ executable

Use LibTorch 2.14.1, CMake 3.18 or newer, and a C++20-capable compiler. The
installed PyTorch package includes the LibTorch headers, libraries, and CMake
configuration, so you can build against the active Python environment:

```bash
python -m pip install cmake
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build cpp/build --config Debug --parallel 2
```

The CMake configuration defaults to Debug and explicitly adds `-g3`, `-O0`,
and `-fno-omit-frame-pointer` for GCC and Clang. It retains unused debug types
and requests complete class definitions with `-femit-class-debug-always`
on GCC or `-fstandalone-debug` on Clang, so GDB can inspect PyTorch types
defined in the headers. These target flags also apply when reusing an older
Release build directory. On Windows it uses `/Zi`,
`/Od`, and `/DEBUG:FULL`. On macOS the build also generates
`cifar10_inference.dSYM`. Keep that bundle alongside the executable when
debugging. The application has full debug information; the prebuilt LibTorch
libraries retain the debug information provided by their distribution.

The CMake setup follows the [official LibTorch installation guide](https://docs.pytorch.org/cppdocs/installing.html).
Build on the machine where you plan to run the executable. On Argon, the GCC
9.5 runtime that fixed Python imports does not by itself establish compiler
compatibility with PyTorch 2.14.1's C++20 headers. Use a compatible newer
compiler/toolchain for this build. If linking against a CUDA-enabled PyTorch
installation, CMake may also require a matching CUDA toolkit; a matching
CPU-only LibTorch distribution avoids that build dependency for this CPU
inference example. For a standalone LibTorch installation, set
`CMAKE_PREFIX_PATH` to its absolute directory instead.

### Run inference on the held-out test set

Prepare the exact test images identified by `split_indices["test"]` in the
training checkpoint, using the cached CIFAR-10 data:

```bash
python prepare_test_images.py
```

This writes all 6,000 held-out images to `data/cifar10_test.bin`. Each record
contains one label byte followed by 3,072 RGB bytes in channel-first order.
`data/cifar10_test.indices.json` records the original dataset indices in file
order. The script checks that the test indices are unique and do not overlap
the training or validation splits. It does not download any data.

Run the C++ executable from this folder:

```bash
./cpp/build/cifar10_inference outputs/cifar10_torchscript.pt \
  data/cifar10_test.bin outputs/cpp_predictions.csv
```

The program loads the TorchScript model and class names with
`torch::jit::load`, uses evaluation mode with gradients disabled, and performs
CPU inference in batches of 128. It converts RGB bytes to float tensors in
NCHW layout and applies the training normalization `(pixel / 255 - 0.5) / 0.5`.
The last batch may contain fewer images. The program prints the first ten
predictions and overall test accuracy, then saves every prediction, actual
class, confidence, and correctness to `outputs/cpp_predictions.csv`.

CSV `image_index` values refer to record order in the binary file. Use the
matching entry in `cifar10_test.indices.json` to find the original CIFAR-10
image. Generated test files, build products, and the predictions CSV are
ignored by Git.

To debug the application on macOS:

```bash
lldb -- ./cpp/build/cifar10_inference outputs/cifar10_torchscript.pt \
  data/cifar10_test.bin outputs/cpp_predictions.csv
```

Then use `breakpoint set --name main --shlib cifar10_inference` and `run`.
On Linux, use the same program arguments with `gdb --args`, followed by
`break main` and `run`.

### Inspect the forward pass with GDB on Argon

For the completed Apptainer workflow, use the container executable at
`cpp/build-apptainer/cifar10_inference` and follow [ARGON.md](ARGON.md).
That guide includes installing GDB when the container does not provide it.

For a native Linux build with GDB already installed, run this from
`cifar10_cnn` in the same environment used to build the executable:

```bash
gdb -x cpp/inference.gdb ./cpp/build/cifar10_inference
```

The command file sets the model, test-image, and output arguments, sets a
breakpoint at `main.cpp:91`, and runs the program. It stops before this line:

```cpp
auto logits = model.forward({images}).toTensor();
```

Inspect the current batch at the GDB prompt:

```text
(gdb) list
(gdb) print start
(gdb) print end
(gdb) print total
(gdb) backtrace
(gdb) next
```

At the first forward pass, `start` should be 0, `end` 128, and `total` 6000.
`next` executes the forward call and advances to the prediction line, where
`logits` has been assigned. `continue` runs to the next batch's breakpoint.
To finish without stopping at each batch, use `disable 1`, then `continue`,
entering each command separately at the GDB prompt.
Debugger-run predictions go to `outputs/cpp_predictions_gdb.csv`.
If `main.cpp` changes, update the breakpoint line in `cpp/inference.gdb`.
