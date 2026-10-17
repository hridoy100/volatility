# Run from cifar10_cnn: gdb -x cpp/inference.gdb ./cpp/build/cifar10_inference
set pagination off
set print pretty on
set args outputs/cifar10_torchscript.pt data/cifar10_test.bin outputs/cpp_predictions_gdb.csv

# Stop before logits is assigned by the first model forward pass.
break main.cpp:91
run
