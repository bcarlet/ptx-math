cmake_minimum_required(VERSION 3.9)

project(model LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main
  main.cu
  cuda_check.cu
  cuda_check.h
)
