add_subdirectory(driver)

if(PTXM_ENABLE_CUDA)
  add_subdirectory(validation)
endif()
