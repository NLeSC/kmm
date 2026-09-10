file(GLOB_RECURSE sources
    "${PROJECT_SOURCE_DIR}/test/*.cpp"
    "${PROJECT_SOURCE_DIR}/test/*.cu")

add_executable(kmmTest ${sources})

if(KMM_USE_HIP)
    set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)
endif()

target_compile_features(kmmTest PRIVATE cxx_std_17)
target_link_libraries(kmmTest PRIVATE kmm)
target_link_libraries(kmmTest PRIVATE Catch2::Catch2WithMain)

find_package(Threads REQUIRED)
target_link_libraries(kmmTest PRIVATE Threads::Threads)

include(Catch)
