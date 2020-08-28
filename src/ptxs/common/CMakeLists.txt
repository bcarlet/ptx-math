target_sources(ptxs PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/bitmask.h
  ${CMAKE_CURRENT_SOURCE_DIR}/fpassert.h
  ${CMAKE_CURRENT_SOURCE_DIR}/fpmanip.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bitcast.c
  ${CMAKE_CURRENT_SOURCE_DIR}/bitcast.h
  ${CMAKE_CURRENT_SOURCE_DIR}/nan.c
  ${CMAKE_CURRENT_SOURCE_DIR}/nan.h
  ${CMAKE_CURRENT_SOURCE_DIR}/squarer.c
  ${CMAKE_CURRENT_SOURCE_DIR}/squarer.h
)

target_compile_options(ptxs PRIVATE $<$<C_COMPILER_ID:GNU>:-mbmi2>)
