add_library(tuning STATIC
  bias.cpp
  bias.hpp
  binsearch.hpp
  coefficients.cpp
  coefficients.hpp
  common.hpp
  counters.cpp
  counters.hpp
)

target_compile_features(tuning PUBLIC cxx_std_11)
