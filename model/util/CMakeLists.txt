add_library(util STATIC
  cuda.cu
  cuda.hpp
  devices.cu
  devices.hpp
  progress.cpp
  progress.hpp
  stopwatch.hpp
)

target_compile_features(util PUBLIC cxx_std_11)

set_target_properties(util PROPERTIES
  CXX_EXTENSIONS OFF
  CMAKE_CUDA_STANDARD 11
)
