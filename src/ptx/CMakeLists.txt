add_library(ptx INTERFACE)

target_sources(ptx INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/ptx.cuh
)

target_compile_features(ptx INTERFACE cxx_std_11)
