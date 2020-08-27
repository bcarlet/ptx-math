add_library(tuning STATIC
  basic_counters.cpp
  basic_counters.hpp
  bias.cpp
  bias.hpp
  coefficients.cpp
  coefficients.hpp
  test_handler.cpp
  test_handler.hpp
)

add_subdirectory(algorithm)
add_subdirectory(base)

target_compile_features(tuning PUBLIC cxx_std_11)
set_target_properties(tuning PROPERTIES CXX_EXTENSIONS OFF)
