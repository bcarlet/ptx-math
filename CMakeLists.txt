cmake_minimum_required(VERSION 3.10)

project(ptx-semantics LANGUAGES C CXX)

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build as a shared library" OFF)
option(PTXS_USE_PIC "Generate position-independent code" OFF)
option(PTXS_ENABLE_CUDA "Enable targets requiring CUDA" ON)

if(PTXS_ENABLE_CUDA)
  enable_language(CUDA)
endif()

if(MSVC)
  set(BASE_OPTS /W4)
else()
  set(BASE_OPTS -Wall -Wextra)
endif()

foreach(OPT IN LISTS BASE_OPTS)
  list(APPEND CUDA_OPTS -Xcompiler=${OPT})
endforeach(OPT)

add_compile_options(
  "$<$<COMPILE_LANGUAGE:C>:${BASE_OPTS}>"
  "$<$<COMPILE_LANGUAGE:CXX>:${BASE_OPTS}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTS}>"
)

add_subdirectory(apps)
add_subdirectory(src)
