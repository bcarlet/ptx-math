add_subdirectory(ptxm)
add_subdirectory(tuning)

if(PTXM_ENABLE_CUDA)
  add_subdirectory(ptx)
  add_subdirectory(util)
endif()
