add_executable(validate
  main.cu
)

target_compile_features(validate PRIVATE cxx_std_11)

target_include_directories(validate PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(validate PRIVATE ptx ptxs util)

install(TARGETS validate DESTINATION bin)
