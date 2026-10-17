# Argon: Apptainer build and GDB

The C++ inference executable was built and inspected with GDB inside
Apptainer on Argon. The executable for this workflow is
`cpp/build-apptainer/cifar10_inference`. The native build uses `cpp/build`
instead, so use the path that matches the build you actually made.

The [README](README.md) covers the four assignment steps, training, model
export, and test-image preparation. This guide covers the container build
and the successful debugger session.

## Start the container

Use an allocated Argon compute session and a container with a C++20-capable
compiler, CMake 3.24 or newer for `--fresh`, and PyTorch/LibTorch compatible
with the exported model. A CUDA-enabled LibTorch installation may also need
the corresponding CUDA toolkit for CMake configuration. The successful
container build reported CMake 3.30.4.

The image used for the successful run was:

```text
/old_Users/smraalam/Projects/volatility/cifar10_cnn/pytorch-cuda12.6.sif
```

Its source was `nvcr.io/nvidia/pytorch:24.10-py3`. To create the image again
when it is not already present, run these commands outside Apptainer:

```bash
cd /old_Users/smraalam/Projects/volatility/cifar10_cnn
unset LD_LIBRARY_PATH LD_PRELOAD
apptainer pull pytorch-cuda12.6.sif \
  docker://nvcr.io/nvidia/pytorch:24.10-py3
```

Launch it from the `cifar10_cnn` directory on the host:

```bash
cd /old_Users/smraalam/Projects/volatility/cifar10_cnn
unset LD_LIBRARY_PATH LD_PRELOAD
cifar_image="/old_Users/smraalam/Projects/volatility/cifar10_cnn/pytorch-cuda12.6.sif"
apptainer shell --cleanenv --nv \
  --bind "$PWD:/workspace" \
  --pwd /workspace \
  "$cifar_image"
```

This [bind mount](https://apptainer.org/docs/user/latest/bind_paths_and_mounts.html)
makes the project available as `/workspace` inside the container. Files
written there remain in the host project directory. Apptainer normally
mounts your home directory as well, which makes the Micromamba installation
below available. The `--nv` option enables the GPU check below.

Run the remaining commands inside the container, from `/workspace`. The
`Apptainer>` text shown by the shell is a prompt, not part of a command.

## Check the container environment

The container was checked directly and reported
`2.5.0a0+e000cf0ad9.nv24.10`, `torch.version.git_version = Unknown`, and
`/usr/local/lib/python3.10/dist-packages/torch/__init__.py`. The
[object and weight inspection guide](GDB_INSPECTION.md) identifies the
matching upstream commit and provides GDB commands for that layout.

The training run recorded Python 3.10.8 and PyTorch 2.14.1+cu126. The
[NVIDIA 24.10 base image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-10.html)
ships Python 3.10, CUDA 12.6.2, and PyTorch 2.5.0a0+e000cf0ad9. The base
image is therefore an older PyTorch environment than the one used for
training. Its `cuda12.6` filename describes the CUDA family, not the PyTorch
version. Record the actual versions inside the container before reproducing
the build; pulling the base image does not reproduce any packages added in
later sessions:

```bash
python -c 'import sys, torch; print(sys.version); print("PyTorch:", torch.__version__); print("CUDA runtime:", torch.version.cuda)'
cmake --version
c++ --version
```

The successful GPU check was reported on an NVIDIA A100 80GB PCIe, with
CUDA available and matrix multiplication working. To repeat that check in
the allocated GPU session:

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
print("GPU:", torch.cuda.get_device_name(0))
matrix = torch.randn(256, 256, device="cuda")
result = matrix @ matrix
torch.cuda.synchronize()
print("GPU matrix multiplication succeeded:", tuple(result.shape))
PY
```

This checks GPU access for the container. The C++ inference example loads
the model with `torch::kCPU` and evaluates the test images on the CPU.

## Prepare the inputs

Use the trained checkpoint and cached dataset from the completed training
run. If the exported model and test binary already exist, reuse them.
Otherwise, run:

```bash
python export_model.py --checkpoint outputs/best_model.pt \
  --output outputs/cifar10_torchscript.pt
python prepare_test_images.py
```

The required inputs are `outputs/cifar10_torchscript.pt` and
`data/cifar10_test.bin`. The binary contains the exact 6,000 held-out test
images selected by the checkpoint's split indices.

## Build with debugging information

Configure using the PyTorch installation inside the container:

```bash
cd /workspace
export TORCH_CUDA_ARCH_LIST="8.0"
cmake --fresh -S cpp -B cpp/build-apptainer \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
```

The architecture setting targets the A100's compute capability, 8.0, during
CUDA configuration. It does not change the C++ program's CPU inference mode.
Use the appropriate architecture when building for a different GPU.

After configuration succeeds, compile:

```bash
cmake --build cpp/build-apptainer -j4
```

The successful build ended with:

```text
[100%] Linking CXX executable cifar10_inference
[100%] Built target cifar10_inference
```

The project's CMake configuration adds `-g3 -O0 -fno-omit-frame-pointer`
for GCC and Clang, plus flags that retain unused header-defined types and
complete class descriptions. This provides application debug information
without optimization. See [GDB_INSPECTION.md](GDB_INSPECTION.md) for the
incomplete-type errors that required these additional flags. Debug
information inside the prebuilt LibTorch libraries depends on how those
libraries were distributed.

For an ordinary inference run in the same container:

```bash
./cpp/build-apptainer/cifar10_inference \
  outputs/cifar10_torchscript.pt \
  data/cifar10_test.bin \
  outputs/cpp_predictions.csv
```

## Make GDB available

The container used for the run did not include GDB: invoking `gdb` produced
`bash: gdb: command not found`. GDB was made available through a separate
Micromamba environment in the user's home directory.

If Micromamba is already at `$HOME/.local/bin/micromamba`, skip this download.
Otherwise, install it once using the
[official Linux download](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):

```bash
mkdir -p "$HOME/.local/bin"
curl -fL https://micro.mamba.pm/api/micromamba/linux-64/latest \
  -o "$HOME/.local/micromamba.tar.bz2"
tar -xjf "$HOME/.local/micromamba.tar.bz2" \
  -C "$HOME/.local" bin/micromamba
```

Create the debugger environment once, inside the container:

```bash
"$HOME/.local/bin/micromamba" create -y \
  -r "$HOME/.local/share/mamba" \
  -p "$HOME/.local/argon-gdb" \
  --override-channels -c conda-forge gdb
```

This installs the [conda-forge GDB package](https://anaconda.org/conda-forge/gdb)
in a separate environment. In later container sessions, reuse the environment
with the launch command below.

## Stop at the model's forward pass

Start GDB inside the container:

```bash
"$HOME/.local/bin/micromamba" run \
  -r "$HOME/.local/share/mamba" \
  -p "$HOME/.local/argon-gdb" \
  gdb -x cpp/inference.gdb \
  ./cpp/build-apptainer/cifar10_inference
```

If GDB is already installed in the image, the equivalent command is:

```bash
gdb -x cpp/inference.gdb ./cpp/build-apptainer/cifar10_inference
```

The command file sets all three program arguments, places a breakpoint at
`main.cpp:91`, and runs the executable. The arguments are:

```text
outputs/cifar10_torchscript.pt data/cifar10_test.bin outputs/cpp_predictions_gdb.csv
```

For manual setup, omit `-x cpp/inference.gdb` from the launch command. Then
enter these commands one at a time at the GDB prompt; do not type `(gdb)`:

```text
(gdb) set args outputs/cifar10_torchscript.pt data/cifar10_test.bin outputs/cpp_predictions_gdb.csv
(gdb) break main.cpp:91
(gdb) run
```

The breakpoint is on this forward call:

```cpp
auto logits = model.forward({images}).toTensor();
```

Use `list`, `print start`, `print end`, and `print total` to inspect the
source and batch bounds. For the first batch of the prepared test set,
the expected bounds are `start = 0`, `end = 128`, and `total = 6000`.
Use `bt` to inspect the call stack, then `next` to execute the forward pass.

The completed session showed the following, with addresses omitted:

```text
Thread 1 "cifar10_inferen" hit Breakpoint 1, main (argc=4, argv=...)
    at /workspace/cpp/main.cpp:91
91              auto logits = model.forward({images}).toTensor();
(gdb) bt
#0  main (argc=4, argv=...) at /workspace/cpp/main.cpp:91
(gdb) next
92              auto predictions = logits.argmax(1);
```

This confirms that GDB stopped before the forward call and that stepping
over it completed the call and advanced to prediction calculation. The
`New Thread` messages seen while running are expected for PyTorch's worker
threads.

## Finish the run

The breakpoint is inside the batch loop. To finish the remaining batches,
first enter:

```gdb
disable 1
```

Press Enter and wait for the next `(gdb)` prompt. Then enter:

```gdb
continue
```

Enter these separately. In the completed session, pasted commands were
interpreted as `disable 1 continue`, causing
`Bad breakpoint number 'continue'`. Entering them separately resolved it.

The program writes `outputs/cpp_predictions_gdb.csv` and prints its final
accuracy when inference finishes. After the program exits, enter `quit` to
leave GDB. The Argon run was reported to complete using this workflow.

## Troubleshooting from the setup

| Symptom | Resolution |
| --- | --- |
| `gdb: command not found` inside Apptainer | Install GDB in the separate environment above and launch it with `micromamba run`. |
| `./cpp/build/cifar10_inference: No such file or directory` | Use `./cpp/build-apptainer/cifar10_inference` for the container build. |
| `No symbol table is loaded` | Load the compiled executable, not the `.pt` file. Confirm the Debug build completed before setting a breakpoint. |
| `Bad breakpoint number 'continue'` | Enter `disable 1` and `continue` at separate GDB prompts. |
| Breakpoint line no longer matches the forward call | Find `model.forward` in `cpp/main.cpp` and update `cpp/inference.gdb`. |

During native Argon setup, adding the entire toolchain
`/cvmfs/argon.hpc.uiowa.edu/2022.2/x-tools/x86_64-unknown-linux-gnu/sysroot/lib`
directory to `LD_LIBRARY_PATH` caused compiler startup failures. Clearing
`LD_LIBRARY_PATH` and `LD_PRELOAD` with the shell's built-in `unset` restored
`g++ --version`. Running `env -u ...` was insufficient as a diagnostic because
`env` itself could fail before removing the variables.

For the separate native Python environment, a user-installed GCC 13
environment at `$HOME/.local/argon-cxx` provided a working compiler and C++
runtime. With `LD_LIBRARY_PATH` set to that environment's `lib` directory,
both GCC 13.4.0 and PyTorch 2.14.1+cu126 loaded successfully. The final C++
build and GDB inspection were completed inside Apptainer as described above.

The dataset, container build directory, prediction CSVs, and `.sif` images
are ignored by Git. Keep the container image outside the repository or at
a stable ignored location for later sessions.
