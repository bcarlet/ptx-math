cmake_minimum_required(VERSION 3.12)

project(ptx-semantics LANGUAGES C CXX CUDA)

if(MSVC)
  set(BASE_OPTS /W4)
else()
  set(BASE_OPTS -Wall -Wextra)
endif()

set(CUDA_OPTS ${BASE_OPTS})
list(TRANSFORM CUDA_OPTS PREPEND -Xcompiler=)

add_compile_options(
  "$<$<COMPILE_LANGUAGE:C>:${BASE_OPTS}>"
  "$<$<COMPILE_LANGUAGE:CXX>:${BASE_OPTS}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTS}>"
)

add_subdirectory(apps)
add_subdirectory(src)
