cmake_minimum_required(VERSION 3.10)
project(cugraphrag LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# CPU-only build: only compile the Python binding
pybind11_add_module(cugraphrag
    python/binding.cpp
)

message(STATUS "CPU-only build mode: CUDA not required")
