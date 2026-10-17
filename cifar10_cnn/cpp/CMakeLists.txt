cmake_minimum_required(VERSION 3.18)
project(cifar10_inference LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)

add_executable(cifar10_inference main.cpp)
target_link_libraries(cifar10_inference PRIVATE "${TORCH_LIBRARIES}")
target_compile_features(cifar10_inference PRIVATE cxx_std_20)
set_target_properties(cifar10_inference PROPERTIES CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Keep full debug information even when an existing build directory used Release.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cifar10_inference PRIVATE -g3 -O0 -fno-omit-frame-pointer)
elseif(MSVC)
    target_compile_options(cifar10_inference PRIVATE /Zi /Od)
    target_link_options(cifar10_inference PRIVATE /DEBUG:FULL)
endif()

if(APPLE)
    find_program(DSYMUTIL_EXECUTABLE dsymutil REQUIRED)
    add_custom_command(TARGET cifar10_inference POST_BUILD
        COMMAND "${DSYMUTIL_EXECUTABLE}" "$<TARGET_FILE:cifar10_inference>"
        VERBATIM)
endif()
