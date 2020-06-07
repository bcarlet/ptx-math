add_library(model STATIC
  model.h
  fpassert.h
  fpmanip.h
  fpmanip.c
  nan.h
  nan.c
  rcp.c
  sqrt.c
  rsqrt.c
  sin.c
  cos.c
  lg2.c
  ex2.c
)

target_compile_features(model PUBLIC c_std_11)
set_target_properties(model PROPERTIES C_EXTENSIONS OFF)
