add_subdirectory(ptxs)
add_subdirectory(tuning)

if(PTXS_ENABLE_CUDA)
  add_subdirectory(ptx)
  add_subdirectory(util)
endif()
