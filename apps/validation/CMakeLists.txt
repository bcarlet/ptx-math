add_executable(ptxmvalidate
  main.cu
)

target_compile_features(ptxmvalidate PRIVATE cxx_std_11)

target_include_directories(ptxmvalidate PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ptxmvalidate PRIVATE ptx ptxm util)

install(TARGETS ptxmvalidate DESTINATION bin)
