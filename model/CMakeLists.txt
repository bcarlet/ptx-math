cmake_minimum_required(VERSION 3.9)

project(model LANGUAGES C CXX CUDA)

add_subdirectory(model)
add_subdirectory(ptx)
add_subdirectory(util)
add_subdirectory(tuning)
add_subdirectory(validation)
