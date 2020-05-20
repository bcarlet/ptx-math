cmake_minimum_required(VERSION 3.9)

project(model LANGUAGES CXX CUDA)

add_executable(main
  main.cu
  cuda_check.cu
  cuda_check.hpp
)
