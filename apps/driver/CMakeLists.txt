add_executable(ptxmrun
  main.cpp
)

target_compile_features(ptxmrun PRIVATE cxx_std_11)

target_include_directories(ptxmrun PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ptxmrun PRIVATE ptxm)

install(TARGETS ptxmrun DESTINATION bin)
