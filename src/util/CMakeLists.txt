add_library(util STATIC
  cuda.cu
  cuda.cuh
  devices.cu
  devices.cuh
  progbar.cpp
  progbar.hpp
  pun.hpp
  stopwatch.hpp
)

target_compile_features(util PUBLIC cxx_std_11)

set_target_properties(util PROPERTIES
  CXX_EXTENSIONS OFF
  CMAKE_CUDA_STANDARD 11
)
