# Run from /workspace inside the container. See GDB_INSPECTION.md.
set logging file outputs/gdb_inspection.log
set logging overwrite off
set logging enabled on
source cpp/torch_inspect.py
source cpp/inference.gdb

# The inferior is the inference program, not GDB or a shell subprocess.
info inferiors
python print("Inference PID:", gdb.selected_inferior().pid)

printf "\nStack objects and fields (hex):\n"
print /x &model
print /x &model._ivalue_
print /x &images
print /x &images.impl_

printf "\nInput tensor metadata and allocation:\n"
torch-tensor images

printf "\nRoot module attributes:\n"
set $root = model._ivalue_.target_
torch-slots $root
