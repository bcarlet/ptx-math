add_library(ptx INTERFACE)

target_sources(ptx INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/ptx.cuh
)

target_include_directories(ptx INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

target_compile_features(ptx INTERFACE cxx_std_11)
