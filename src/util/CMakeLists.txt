add_library(util STATIC
  cuda.cuh
  progbar.cpp
  progbar.hpp
  pun.hpp
  stopwatch.hpp
)

target_include_directories(util INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

target_compile_features(util PUBLIC cxx_std_11)
