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
and no `ClassType` debug definition. After rebuilding with the flags below,
the remote session successfully printed tensor metadata and root module
attributes. If a field is missing, inspect its type instead of guessing
a byte offset.

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
```

```gdb
python print("Inference PID:", gdb.selected_inferior().pid)
```

```gdb
p/x &model
```

```gdb
p/x &model._ivalue_
```

```gdb
p/x &images
```

```gdb
p/x &images.impl_
```

This prints the [inferior's PID](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Inferiors-In-Python.html),
not the PID of GDB or a shell subprocess. All addresses come from the live
process. They can change between runs.

## Inspect the input tensor

```gdb
ptype /o images
```

```gdb
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
```

```gdb
p/x $input_data
```

```gdb
x/8fw $input_data
```

```gdb
x/8xw $input_data
```

`$torch_data` is the allocation base. `$torch_offset` is measured in tensor
elements, not bytes. Apply the offset before reading values; strides also
matter for noncontiguous tensors. The helper does not read GPU data.

## Follow the module hierarchy and layer descriptions

```gdb
set $root = model._ivalue_.target_
```

```gdb
p/x $root
```

```gdb
ptype /o *$root
```

```gdb
torch-slots $root
```

```gdb
torch-attr $root features
```

```gdb
set $features = $torch_object
```

```gdb
torch-slots $features
```

```gdb
torch-attr $features 0
```

```gdb
set $conv = $torch_object
```

```gdb
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
```

```gdb
p $torch_type->attributes_
```

```gdb
p $torch_type->constantNames_
```

```gdb
p $torch_type->constantValues_
```

```gdb
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
```

```gdb
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
```

```gdb
ptype /o c10::ClassType
```

Both should show full definitions with nonzero sizes. Then rerun
`torch-tensor images`, set `$root = model._ivalue_.target_`, and run
`torch-slots $root`. The PID and addresses will belong to the new process;
print and record them again. The rebuilt Argon session verified the tensor
and root module inspection, as recorded below.

## Verified Argon input and root inspection

After the clean rebuild, GDB stopped at `main.cpp:91` in inference process
**37609**. The provided transcript recorded these addresses in that process:

| Object or field | Hexadecimal address |
| --- | --- |
| Model wrapper and its first field, `_ivalue_` | `0x7fffffffd7d0` |
| Input tensor wrapper and its first field, `impl_` | `0x7fffffffd710` |
| Input `TensorImpl` | `0x5555591ec210` |
| Its `storage_` field | `0x5555591ec220` |
| Input `StorageImpl` | `0x55555917f500` |
| Its `data_ptr_` field | `0x55555917f510` |
| Input allocation base | `0x555559760000` |
| Root module runtime `Object` | `0x555559180fc0` |
| Root `ClassType` | `0x555559153750` |
| Root `slots_` field | `0x555559181010` |

The input had four dimensions, shape `[128, 3, 32, 32]`, strides
`[3072, 1024, 32, 1]`, storage offset zero, and 393,216 elements. The device
metadata identified the CPU.

GDB printed both members of the shape/stride union. In this case the active
member is `inlineStorage_`, because four dimensions fit within its five
inline dimension slots. The apparent `outOfLineStorage_ = 0x80` is the
overlapping representation of the first dimension, 128; it is not the
numeric data allocation address. See the matching
[SizesAndStrides.h](https://github.com/pytorch/pytorch/blob/e000cf0ad980e5d140dc895a646174e9b945cf26/c10/core/impl/SizesAndStrides.h).

The root object had these slots:

| Slot | Name | Kind | Value tag |
| --- | --- | --- | --- |
| 0 | `training` | Regular attribute | Bool |
| 1 | `_is_full_backward_hook` | Regular attribute | None |
| 2 | `features` | Regular attribute | Object |
| 3 | `classifier` | Regular attribute | Object |

These results verify that the descriptions and values can now be paired
in the live process. A follow-up inspection reached the first convolution
and its weight allocation, as recorded below. These addresses are
historical observations for PID 37609, not expected addresses for a later
run.

## Locate weights and read sample values

The follow-up output for `features.0` recorded:

| Object or field | Hexadecimal address |
| --- | --- |
| Convolution runtime object | `0x5555591ea6b0` |
| Convolution `ClassType` | `0x5555591c65e0` |
| Weight attribute descriptor | `0x5555591c6990` |
| Weight `IValue` slot and tensor wrapper | `0x55555915df20` |
| Weight `TensorImpl` | `0x5555591b9430` |
| Its `storage_` field | `0x5555591b9440` |
| Weight `StorageImpl` | `0x555559184f90` |
| Its `data_ptr_` field | `0x555559184fa0` |
| Weight allocation base | `0x5555591a91c0` |

The weight was a CPU tensor with shape `[32, 3, 3, 3]`, strides
`[27, 9, 3, 1]`, storage offset zero, and 864 elements. The descriptor
identified it as a parameter; the numeric weights reside in the allocation
reached through the tensor and storage pointers. A subsequent memory dump
at `0x5555591a91c0` showed these first eight weights:

| Element | Float32 value printed by GDB | Hexadecimal word |
| --- | ---: | --- |
| 0 | 0.119935118 | `0x3df5a08b` |
| 1 | 0.291540712 | `0x3e9544d3` |
| 2 | -0.0696675777 | `0xbd8eade0` |
| 3 | 0.142484173 | `0x3e11e75f` |
| 4 | 0.0102107441 | `0x3c274af7` |
| 5 | -0.0470474996 | `0xbd40b4e1` |
| 6 | -0.182749048 | `0xbe3b2291` |
| 7 | 0.0387827568 | `0x3d1edaab` |

Each hexadecimal word is a float's 32-bit encoding, not another pointer.
The eight encodings were checked against the reported decimal values.

```gdb
torch-attr $conv weight
```

```gdb
torch-tensor *$torch_tensor
```

```gdb
set $weight_data = (float*)$torch_data + $torch_offset
```

```gdb
p/x $weight_data
```

```gdb
x/8fw $weight_data
```

```gdb
x/8xw $weight_data
```

Record these separate addresses: `$torch_slot` is the owning `IValue`
slot, `$torch_tensor` is the wrapper inside its payload, `$torch_impl` is
the metadata object, and `$torch_data` is the allocation base. The first
convolution's float32 weights have shape `[32, 3, 3, 3]`, or 864 elements;
its bias has 32 elements. Inspect the bias with `torch-attr $conv bias`,
then `torch-tensor *$torch_tensor`.

If printing `constantNames_` only displays vector internals, print its
strings using the helper's memory-reading iterator. This is one GDB command:

```gdb
python for i, n in enumerate(vector_items(gdb.parse_and_eval("$torch_type->constantNames_"))): print(i, n["_M_dataplus"]["_M_p"].string())
```

Each displayed index corresponds to the same index in `constantValues_`.
The observed name-buffer start was `0x5555591b51c0`; the value-buffer start
was `0x5555591c6440`. Those buffers hold descriptions and constant values,
separately from the learned weight allocation.

The live convolution's constant-name order was:

| Index | Name |
| --- | --- |
| 0 | `padding` |
| 1 | `groups` |
| 2 | `output_padding` |
| 3 | `dilation` |
| 4 | `stride` |
| 5 | `kernel_size` |
| 6 | `in_channels` |
| 7 | `padding_mode` |
| 8 | `out_channels` |

For this observed order, read the integer constants `groups`, `in_channels`,
and `out_channels` together:

```gdb
set $constants = $torch_type->constantValues_._M_impl._M_start
```

```gdb
p {$constants[1].payload.u.as_int, $constants[6].payload.u.as_int, $constants[8].payload.u.as_int}
```

The live inspection returned `{1, 3, 32}`, confirming one group, three
input channels, and 32 output channels, matching the model definition.
The brace syntax is a [GDB array expression](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Expressions.html).
Check the tag before interpreting a different constant's union payload:

```gdb
p $constants[5].tag
```

For a `Tuple` tag, the two-element kernel size can be followed through
`c10::ivalue::Tuple`. Its two elements fit in `TupleElements`' inline array:

```gdb
set $kernel = (c10::ivalue::Tuple*)$constants[5].payload.u.as_intrusive_ptr
```

```gdb
p {$kernel->elements_.elementsInline_[0].payload.u.as_int, $kernel->elements_.elementsInline_[1].payload.u.as_int}
```

The expected kernel size is `{3, 3}`. For another layer or archive, list
the names again rather than reusing these constant indices blindly.

Compare with batch normalization, the next layer:

```gdb
torch-attr $features 1
```

```gdb
set $batchnorm = $torch_object
```

```gdb
torch-slots $batchnorm
```

```gdb
torch-attr $batchnorm running_mean
```

```gdb
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
```

```gdb
torch-tensor logits
```

The first batch's output shape is `[128, 10]`. To finish, enter `disable 1`,
wait for the next prompt, then enter `continue`. After the program exits,
enter `set logging enabled off`, then `quit`.

## Verified forward-pass output

In the same recorded session, PID **37609**, `next` advanced from the
forward call at line 91 to `auto predictions = logits.argmax(1);` at line
92. Inspecting `logits` then produced:

| Object or field | Hexadecimal address |
| --- | --- |
| Output tensor wrapper and its `impl_` field | `0x7fffffffd720` |
| Output `TensorImpl` | `0x555559531c10` |
| Its `storage_` field | `0x555559531c20` |
| Output `StorageImpl` | `0x5555595cdc60` |
| Its `data_ptr_` field | `0x5555595cdc70` |
| Output allocation base | `0x55555959e980` |

The output was a CPU tensor with shape `[128, 10]`, strides `[10, 1]`,
storage offset zero, and 1,280 elements. Each of the 128 input images
therefore produced ten class logits. These are the model's raw class
scores; the program calculates predictions and confidence after the
forward pass.

This session verified the breakpoint, inference PID, input and model
addresses, root module organization, first-convolution parameter storage
and sample values, channel/group constants, and the resulting output
tensor. The source links above pin the interpretation to the container's
upstream PyTorch commit. Completion of the remaining batches in this
particular debugger run was not included in the provided transcript.

Retain the log for your report. Record the actual PID, matching source
commit, object and field addresses, module paths, parameter/buffer roles,
weight shapes and samples, and the step from line 91 to line 92. Use
addresses from the same live process for each observation.
