add_library(tuning STATIC
  interval.hpp
  fntypes.hpp
  batching.cpp
  batching.hpp
  binsearch.cpp
  binsearch.hpp
  counters.cpp
  counters.hpp
  search.cpp
  search.hpp
  testing.cpp
  testing.hpp
)

target_compile_features(tuning PUBLIC cxx_std_11)
set_target_properties(tuning PROPERTIES CXX_EXTENSIONS OFF)
