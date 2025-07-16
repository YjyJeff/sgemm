cmake_minimum_required(VERSION 3.10)

project(gemm LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets --ptxas-options=-v")

option(TRANSPOSE "Transpose the left matrix" ON)

if(TRANSPOSE)
    add_definitions(-DTRANSPOSE)
endif()

add_executable(blocked_gemm blocked_gemm.cu)
add_executable(shared_blocked_gemm shared_blocked_gemm.cu)
add_executable(thread_tiling thread_tiling.cu)
add_executable(prefetch prefetch.cu)
add_executable(prefetch_while_loop prefetch_while_loop.cu)
add_executable(prefetch_v_load prefetch_v_load.cu)
