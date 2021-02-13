add_library(tuning STATIC
  batching.cpp
  batching.hpp
  bias.cpp
  bias.hpp
  binsearch.hpp
  coefficients.cpp
  coefficients.hpp
  counter.hpp
  model.hpp
  sign_counter.cpp
  sign_counter.hpp
  testing.cpp
  testing.hpp
)

target_compile_features(tuning PUBLIC cxx_std_11)
