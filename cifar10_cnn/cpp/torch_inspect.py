"""GDB commands for the NVIDIA 24.10 PyTorch build and GNU libstdc++.

Field paths follow pytorch/pytorch commit
e000cf0ad980e5d140dc895a646174e9b945cf26. These commands read debug types
and process memory; they do not call functions in the stopped program.
Load at the forward-pass breakpoint with: source cpp/torch_inspect.py
"""

import gdb


def address(label, pointer):
    if pointer is None:
        gdb.write(f"{label}: GDB temporary, no address in the process\n")
    else:
        gdb.write(f"{label}: 0x{int(pointer):x}\n")


def debug_type(name):
    try:
        result = gdb.lookup_type(name).strip_typedefs()
        complete = result.sizeof > 0
    except gdb.error:
        complete = False
    if not complete:
        raise gdb.GdbError(
            f"Missing complete debug type {name}. Rebuild the executable with "
            "the type-retention flags in GDB_INSPECTION.md, then restart GDB."
        )
    return result


def vector_items(vector):
    start = vector["_M_impl"]["_M_start"]
    finish = vector["_M_impl"]["_M_finish"]
    for index in range(int(finish - start)):
        yield (start + index).dereference()


def model_type(pointer):
    class_type = debug_type("c10::ClassType")
    # This ClassType uses the shared arm of SingletonOrSharedTypePtr.
    shared = pointer.dereference()["type_"]["type_"]["repr_"]["shared_"]["repr_"]
    return shared["_M_ptr"].cast(class_type.pointer())


def attributes(pointer):
    descriptors = vector_items(model_type(pointer).dereference()["attributes_"])
    values = vector_items(pointer.dereference()["slots_"])
    for descriptor, value in zip(descriptors, values):
        name = descriptor["attributeName_"]["_M_dataplus"]["_M_p"].string()
        yield name, descriptor, value


class TorchSlots(gdb.Command):
    """torch-slots OBJECT_POINTER: list named module attributes and their slots."""

    def __init__(self):
        super().__init__("torch-slots", gdb.COMMAND_DATA)

    def invoke(self, argument, from_tty):
        pointer = gdb.parse_and_eval(argument)
        address("Object", pointer)
        class_pointer = model_type(pointer)
        gdb.set_convenience_variable("torch_type", class_pointer)
        address("ClassType", class_pointer)
        address("slots_ field", pointer.dereference()["slots_"].address)
        for index, (name, descriptor, value) in enumerate(attributes(pointer)):
            kind = str(descriptor["kind_"]).rsplit("::", 1)[-1]
            tag = str(value["tag"]).rsplit("::", 1)[-1]
            gdb.write(f"[{index}] {name}: {kind}, {tag}\n")
            address("  descriptor", descriptor.address)
            address("  IValue slot", value.address)


class TorchAttr(gdb.Command):
    """torch-attr OBJECT_POINTER NAME: select an attribute by its actual name.

Sets $torch_slot to its IValue address, $torch_object for an Object attribute,
and $torch_tensor for a Tensor attribute. The latter variables are cleared
when their type does not apply.
"""

    def __init__(self):
        super().__init__("torch-attr", gdb.COMMAND_DATA)

    def invoke(self, argument, from_tty):
        args = gdb.string_to_argv(argument)
        if len(args) != 2:
            raise gdb.GdbError("Usage: torch-attr OBJECT_POINTER NAME")
        pointer = gdb.parse_and_eval(args[0])
        for name, descriptor, value in attributes(pointer):
            if name != args[1]:
                continue
            tag = str(value["tag"]).rsplit("::", 1)[-1]
            gdb.set_convenience_variable("torch_slot", value.address)
            gdb.set_convenience_variable("torch_object", None)
            gdb.set_convenience_variable("torch_tensor", None)
            address(f"{name} IValue slot ($torch_slot)", value.address)
            address("attribute descriptor", descriptor.address)
            gdb.write(f"tag: {tag}\n")
            if tag == "Object":
                child = value["payload"]["u"]["as_intrusive_ptr"].cast(
                    gdb.lookup_type("c10::ivalue::Object").pointer()
                )
                gdb.set_convenience_variable("torch_object", child)
                address("child Object ($torch_object)", child)
            elif tag == "Tensor":
                tensor_pointer = value["payload"]["as_tensor"].address
                gdb.set_convenience_variable("torch_tensor", tensor_pointer)
                address("at::Tensor ($torch_tensor)", tensor_pointer)
            return
        raise gdb.GdbError(f"No attribute named {args[1]!r}")


class TorchTensor(gdb.Command):
    """torch-tensor TENSOR: show the tensor wrapper, metadata, and allocation."""

    def __init__(self):
        super().__init__("torch-tensor", gdb.COMMAND_DATA)

    def invoke(self, argument, from_tty):
        tensor = gdb.parse_and_eval(argument)
        impl = tensor["impl_"]["target_"]
        gdb.set_convenience_variable("torch_data", None)
        gdb.set_convenience_variable("torch_offset", None)
        gdb.set_convenience_variable("torch_impl", impl)
        address("at::Tensor wrapper", tensor.address)
        address("impl_ field", tensor["impl_"].address)
        address("TensorImpl ($torch_impl)", impl)
        if int(impl) == 0:
            raise gdb.GdbError("Expected a defined, storage-backed CPU tensor")
        impl = impl.cast(debug_type("c10::TensorImpl").pointer())
        storage = impl.dereference()["storage_"]["storage_impl_"]["target_"]
        if int(storage) == 0:
            raise gdb.GdbError("Expected a defined, storage-backed CPU tensor")
        storage = storage.cast(debug_type("c10::StorageImpl").pointer())
        data_ptr = storage.dereference()["data_ptr_"]
        data = data_ptr["ptr_"]["data_"]
        gdb.set_convenience_variable("torch_impl", impl)
        gdb.set_convenience_variable("torch_data", data)
        gdb.set_convenience_variable("torch_offset", impl.dereference()["storage_offset_"])
        address("storage_ field", impl.dereference()["storage_"].address)
        address("StorageImpl", storage)
        address("data_ptr_ field", data_ptr.address)
        address("allocation base ($torch_data)", data)
        for field in ("storage_offset_", "numel_", "sizes_and_strides_", "data_type_", "device_opt_"):
            gdb.write(f"{field}:\n")
            gdb.execute(f"print $torch_impl->{field}")


TorchSlots()
TorchAttr()
TorchTensor()
