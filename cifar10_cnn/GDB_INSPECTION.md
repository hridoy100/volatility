# Inspect model objects and weights with GDB

Use the [working Apptainer environment](ARGON.md) and its Debug executable,
`cpp/build-apptainer/cifar10_inference`. Run commands from `/workspace`,
the mounted `cifar10_cnn` directory. Enter each interactive GDB command at
a separate prompt.

## Match the PyTorch source

The container reported:

```text
torch.__version__: 2.5.0a0+e000cf0ad9.nv24.10
torch.version.git_version: Unknown
torch.__file__: /usr/local/lib/python3.10/dist-packages/torch/__init__.py
```

[NVIDIA's release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-10.html)
map this build to upstream commit
[`e000cf0ad980e5d140dc895a646174e9b945cf26`](https://github.com/pytorch/pytorch/commit/e000cf0ad980e5d140dc895a646174e9b945cf26).
Select that commit on GitHub rather than `main`, the `v2.5.0` release tag,
or the training environment's 2.14.1 version. The NVIDIA build is a
pre-release with its own suffix.

The helper's field paths were checked against this commit. It also relies
on GNU libstdc++'s vector and string layouts. NVIDIA may carry additional
patches; the installed headers under
`/usr/local/lib/python3.10/dist-packages/torch/include` and GDB's `ptype /o`
output are the final check for the actual build. The first remote run
printed the PID and wrapper addresses, but found an incomplete `TensorImpl`
and no `ClassType` debug definition. The rebuild instructions below address
that missing type information. If a field is missing, inspect its type
instead of guessing a byte offset.

| Structure | What it holds | Source at the matching commit |
| --- | --- | --- |
| `torch::jit::Module`, derived from `Object` | `_ivalue_` points to the runtime object | [module.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/torch/csrc/jit/api/module.h), [object.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/torch/csrc/jit/api/object.h) |
| `c10::ivalue::Object` | `type_` describes the object; `slots_` holds current attribute values | [ivalue_inl.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/aten/src/ATen/core/ivalue_inl.h) |
| `ClassType`, `ClassAttribute`, and the type pointer wrapper | Names, declared types, parameter/buffer roles, constants, and methods | [class_type.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/aten/src/ATen/core/class_type.h), [jit_type_base.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/aten/src/ATen/core/jit_type_base.h) |
| `c10::IValue` | A `tag` and `payload`; objects and tensors use different union members | [ivalue.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/aten/src/ATen/core/ivalue.h) |
| `at::TensorBase`, `c10::intrusive_ptr`, `TensorImpl` | Wrapper, managed pointer, shape/strides, dtype, device, storage, offset, and element count | [TensorBase.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/aten/src/ATen/core/TensorBase.h), [intrusive_ptr.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/c10/util/intrusive_ptr.h), [TensorImpl.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/c10/core/TensorImpl.h) |
| `Storage`, `StorageImpl`, `DataPtr`, `UniqueVoidPtr` | Storage ownership and the underlying allocation address | [Storage.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/c10/core/Storage.h), [StorageImpl.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/c10/core/StorageImpl.h), [Allocator.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/c10/core/Allocator.h), [UniqueVoidPtr.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/c10/util/UniqueVoidPtr.h) |

## Start at the forward pass and print the PID

Copy the updated `cpp/inspect_model.gdb`, `cpp/torch_inspect.py`, and
`cpp/inference.gdb` files to the Argon project before using them. A missing
`cpp/inspect_model.gdb` means the new scripts have not reached that project;
the executable can still be present and load normally.

From a separate terminal on the Mac, this copies the scripts directly
without requiring a Git commit or push:

```bash
scp -P 40 \
  /Users/smraalam/Downloads/Software/volatility/cifar10_cnn/cpp/inspect_model.gdb \
  /Users/smraalam/Downloads/Software/volatility/cifar10_cnn/cpp/torch_inspect.py \
  /Users/smraalam/Downloads/Software/volatility/cifar10_cnn/cpp/inference.gdb \
  smraalam@argon.hpc.uiowa.edu:/old_Users/smraalam/Projects/volatility/cifar10_cnn/cpp/
```

Use the SSH port from your normal Argon connection. The example uses port
40, as in the [Argon connection instructions](https://uiowa.atlassian.net/wiki/spaces/hpcdocs/pages/540606569/Comparing+IDAS+and+Argon);
change `-P 40` to `-P 22` if you normally use port 22. Alternatively, once
these files are committed and pushed, pull the commit into the Argon
checkout before starting GDB.

Inside the existing Apptainer session, confirm the bind mount sees them:

```bash
ls -l cpp/inspect_model.gdb cpp/torch_inspect.py cpp/inference.gdb
```

The helpers do not change C++ code; start with the existing Debug binary.
Inside Apptainer, launch:

```bash
"$HOME/.local/bin/micromamba" run \
  -r "$HOME/.local/share/mamba" \
  -p "$HOME/.local/argon-gdb" \
  gdb -x cpp/inspect_model.gdb \
  ./cpp/build-apptainer/cifar10_inference
```

If GDB is already on the container's `PATH`, use
`gdb -x cpp/inspect_model.gdb ./cpp/build-apptainer/cifar10_inference`.

The script sets the arguments and stops at `main.cpp:91`, before the
forward call. It then prints the inference PID, wrapper and field addresses
in hexadecimal, input tensor metadata and storage addresses, and root
module attributes. GDB remains paused for further inspection. The output
is appended to `outputs/gdb_inspection.log`.

If already paused at line 91 in your earlier GDB session, keep that process.
Enter `set logging file outputs/gdb_inspection.log`, then
`set logging enabled on`, then `source cpp/torch_inspect.py` as separate
commands. Print the PID and addresses with the commands below.

```gdb
info inferiors
python print("Inference PID:", gdb.selected_inferior().pid)
p/x &model
p/x &model._ivalue_
p/x &images
p/x &images.impl_
```

This prints the [inferior's PID](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Inferiors-In-Python.html),
not the PID of GDB or a shell subprocess. All addresses come from the live
process. They can change between runs.

## Inspect the input tensor

```gdb
ptype /o images
torch-tensor images
```

The helper reads memory and debug types without calling functions in the
stopped program. It follows:

```text
at::Tensor -> impl_.target_ -> TensorImpl
    -> storage_.storage_impl_.target_ -> StorageImpl
    -> data_ptr_.ptr_.data_ -> allocation base
```

It prints both field addresses and pointer values, followed by storage
offset, element count, raw shape/stride fields, dtype metadata, and device
metadata. A wrapper, its pointer field, its metadata object, and its
numeric allocation are distinct observations; multiple tensors may share
one allocation. The first input batch has shape `[128, 3, 32, 32]`, or
393,216 float32 elements on the CPU.

For this input, inspect eight values as floats and as raw hexadecimal words:

```gdb
set $input_data = (float*)$torch_data + $torch_offset
p/x $input_data
x/8fw $input_data
x/8xw $input_data
```

`$torch_data` is the allocation base. `$torch_offset` is measured in tensor
elements, not bytes. Apply the offset before reading values; strides also
matter for noncontiguous tensors. The helper does not read GPU data.

## Follow the module hierarchy and layer descriptions

```gdb
set $root = model._ivalue_.target_
p/x $root
ptype /o *$root
torch-slots $root
torch-attr $root features
set $features = $torch_object
torch-slots $features
torch-attr $features 0
set $conv = $torch_object
torch-slots $conv
```

`torch-slots` pairs attribute descriptors with their runtime slots and
prints their addresses, names, roles, and tags. `torch-attr` selects by
name, so no fixed slot number is assumed. `features` is the sequential
feature extractor; its child `0` is the first convolution. The final
`torch-slots` call sets `$torch_type` to the convolution's `ClassType`.

Inspect the descriptions and behavior stored on this type:

```gdb
p/x $torch_type
p $torch_type->attributes_
p $torch_type->constantNames_
p $torch_type->constantValues_
p $torch_type->methods_
```

Each descriptor contains a name, declared type, and `AttributeKind`:
parameter, buffer, or regular attribute. `weight` and `bias` are parameters.
Layer configuration such as kernel size, padding, and stride may be stored
as constants or attributes; inspect the actual tables. Constants can also
appear in compiled method graphs. `methods_` holds function pointers for
behavior, rather than the weight data.

After checking that `methods_` is nonempty, inspect a listed method:

```gdb
set print object on
p *$torch_type->methods_._M_impl._M_start[0]
```

Internal function/graph details depend on the debug types provided by
LibTorch. Application debug information cannot supply missing symbols for
a stripped prebuilt library. Use the matching source above to interpret
the fields that are available.

The initial remote inspection reported `TensorImpl` as an incomplete type
of size zero and could not find `c10::ClassType`. The numeric pointer was
available, but GDB lacked the structure definitions needed to follow it.

The current CMake configuration retains unused header-defined debug types.
GCC also uses `-femit-class-debug-always`; Clang uses `-fstandalone-debug`.
These settings are needed in addition to `-g3 -O0` when prebuilt libraries
do not provide complete type definitions to the debugger. See the
[GCC options](https://gcc.gnu.org/onlinedocs/gcc/Debugging-Options.html) and
[Clang debug options](https://clang.llvm.org/docs/UsersManual.html#controlling-debug-information).

To apply the fix immediately to an existing GCC build on Argon without
first transferring the updated CMake file, quit GDB and reconfigure inside
the same container:

```bash
cmake -S cpp -B cpp/build-apptainer \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS_DEBUG="-g3 -O0 -fno-eliminate-unused-debug-types -femit-class-debug-always" \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build cpp/build-apptainer --clean-first -j4
```

This adds type information from application headers. It does not rebuild
LibTorch or add debug information for implementation types absent from
those headers.

Restart GDB on the rebuilt executable; the old paused process still uses
the old binary and debug information. At the breakpoint, verify:

```gdb
ptype /o c10::TensorImpl
ptype /o c10::ClassType
```

Both should show full definitions with nonzero sizes. Then rerun
`torch-tensor images`, set `$root = model._ivalue_.target_`, and run
`torch-slots $root`. The PID and addresses will belong to the new process;
print and record them again. The rebuilt remote inspection still needs
verification in Argon's GDB.

## Locate weights and read sample values

```gdb
torch-attr $conv weight
torch-tensor *$torch_tensor
set $weight_data = (float*)$torch_data + $torch_offset
p/x $weight_data
x/8fw $weight_data
x/8xw $weight_data
```

Record these separate addresses: `$torch_slot` is the owning `IValue`
slot, `$torch_tensor` is the wrapper inside its payload, `$torch_impl` is
the metadata object, and `$torch_data` is the allocation base. The first
convolution's float32 weights have shape `[32, 3, 3, 3]`, or 864 elements;
its bias has 32 elements. Inspect the bias with `torch-attr $conv bias`,
then `torch-tensor *$torch_tensor`.

Compare with batch normalization, the next layer:

```gdb
torch-attr $features 1
set $batchnorm = $torch_object
torch-slots $batchnorm
torch-attr $batchnorm running_mean
torch-tensor *$torch_tensor
```

Its `weight` and `bias` are parameters. `running_mean`, `running_var`, and
`num_batches_tracked` are buffers. The last is an integer tensor, so do
not cast its data to float32. The classifier is available through
`torch-attr $root classifier`; children `1` and `4` are its linear layers.

## Execute forward and finish

The program is still paused before the original forward call. Enter:

```gdb
next
```

At line 92, `logits` has been initialized. Inspect its output allocation:

```gdb
p/x &logits
torch-tensor logits
```

The first batch's output shape is `[128, 10]`. To finish, enter `disable 1`,
wait for the next prompt, then enter `continue`. After the program exits,
enter `set logging enabled off`, then `quit`.

Retain the log for your report. Record the actual PID, matching source
commit, object and field addresses, module paths, parameter/buffer roles,
weight shapes and samples, and the step from line 91 to line 92. Use
addresses from the same live process for each observation.
