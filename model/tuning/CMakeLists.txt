add_library(tuning STATIC
  bias.cpp
  bias.hpp
)

add_subdirectory(algorithm)
add_subdirectory(base)

target_compile_features(tuning PUBLIC cxx_std_11)
set_target_properties(tuning PROPERTIES CXX_EXTENSIONS OFF)
