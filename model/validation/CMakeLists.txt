add_executable(validate
  main.cu
)

target_compile_features(validate PRIVATE cxx_std_11)

set_target_properties(validate PROPERTIES
  CXX_EXTENSIONS OFF
  CMAKE_CUDA_STANDARD 11
)

target_include_directories(validate PRIVATE ../)
target_link_libraries(validate PRIVATE model ptx util)
