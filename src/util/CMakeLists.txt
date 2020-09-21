add_library(util STATIC
  cuda.cu
  cuda.cuh
  progbar.cpp
  progbar.hpp
  pun.hpp
  stopwatch.hpp
)

target_compile_features(util PUBLIC cxx_std_11)
