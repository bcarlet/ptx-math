cmake_minimum_required(VERSION 3.9)

project(model LANGUAGES C CXX CUDA)

add_executable(main
  main.cu
  cuda_util.cu
  cuda_util.hpp
  ptx.hpp
)

add_subdirectory(util)

target_compile_features(main PRIVATE cxx_std_11)
target_compile_options(main PRIVATE "$<$<CONFIG:RELEASE>:-O2>")
