# Every example is a .cu file with real GPU kernels, so it can only be compiled by nvcc/hipcc.
if(KMM_USE_CUDA OR KMM_USE_HIP)
  add_subdirectory(hello_world)
  add_subdirectory(vector_add)
  add_subdirectory(reduce_sum)
endif()
