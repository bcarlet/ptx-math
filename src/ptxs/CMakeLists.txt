add_library(ptxs STATIC
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

target_compile_features(ptxs PUBLIC c_std_11)
set_target_properties(ptxs PROPERTIES C_EXTENSIONS OFF)
