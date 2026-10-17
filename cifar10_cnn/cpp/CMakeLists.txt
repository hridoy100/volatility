cmake_minimum_required(VERSION 3.18)
project(cifar10_inference LANGUAGES CXX)

find_package(Torch REQUIRED)

add_executable(cifar10_inference main.cpp)
target_link_libraries(cifar10_inference PRIVATE "${TORCH_LIBRARIES}")
target_compile_features(cifar10_inference PRIVATE cxx_std_20)
set_target_properties(cifar10_inference PROPERTIES CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
