add_library(ptxm
  models.h
  tuning.h
  params.h
  rcp.c
  sqrt.c
  rsqrt.c
  sincos.c
  lg2.c
  ex2.c
)

add_subdirectory(common)
add_subdirectory(reduction)
add_subdirectory(tables)

target_include_directories(ptxm INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

target_compile_features(ptxm PUBLIC c_std_11)
set_target_properties(ptxm PROPERTIES C_EXTENSIONS OFF)

if(NOT MSVC)
  target_compile_options(ptxm PRIVATE -mbmi2)
endif()

if(PTXM_USE_PIC)
  set_target_properties(ptxm PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

install(TARGETS ptxm DESTINATION lib)
install(FILES models.h DESTINATION include/ptxm)
