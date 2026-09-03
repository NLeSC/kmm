add_executable(hello_world ${CMAKE_CURRENT_SOURCE_DIR}/main.cu)

if(KMM_USE_CUDA)
    set_target_properties(
            hello_world
            PROPERTIES
            CUDA_ARCHITECTURES "80"
    )
endif()

if(KMM_USE_HIP)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/main.cu PROPERTIES LANGUAGE HIP)
endif()

target_compile_features(hello_world PRIVATE cxx_std_17)
target_link_libraries(hello_world PRIVATE kmm)
