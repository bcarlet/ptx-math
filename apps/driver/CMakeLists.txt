add_executable(ptxmrun
  main.cpp
)

target_compile_features(ptxmrun PRIVATE cxx_std_11)
target_link_libraries(ptxmrun PRIVATE ptxm)

install(TARGETS ptxmrun DESTINATION bin)
