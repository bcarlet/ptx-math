add_library(tuning STATIC
  bias.cpp
  bias.hpp
  binsearch.hpp
  coefficients.cpp
  coefficients.hpp
  common.hpp
)

target_include_directories(tuning INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

target_compile_features(tuning PUBLIC cxx_std_11)
