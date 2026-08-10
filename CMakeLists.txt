cmake_minimum_required(VERSION 3.18)

project(kmm VERSION 0.1.0 LANGUAGES CXX)

option(KMM_BUILD_TESTS "Build kmm tests" ON)
option(KMM_USE_CUDA "Enable CUDA support" OFF)

file(GLOB_RECURSE sources src/*.cpp)

add_library(kmm ${sources})

target_include_directories(kmm PUBLIC ${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_features(kmm PUBLIC cxx_std_17)
target_compile_options(kmm PRIVATE
    -Wall -Wextra -Wconversion -Wno-unused-parameter
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
)

if(KMM_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set_target_properties(kmm PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )

    file(GLOB_RECURSE cuda_sources src/*.cu)
    target_sources(kmm PRIVATE ${cuda_sources})

    target_link_libraries(kmm PUBLIC CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(kmm PUBLIC KMM_USE_CUDA)
endif()


# Enable PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Dependencies: fmt
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
add_subdirectory(external/fmt)
target_link_libraries(${PROJECT_NAME} PUBLIC fmt)

# Dependencies: spdlog
set(SPDLOG_FMT_EXTERNAL ON)
set(SPDLOG_BUILD_PIC ON)
add_subdirectory(external/spdlog)
target_link_libraries(${PROJECT_NAME} PUBLIC spdlog)

# Dependencies: unordered_dense
add_subdirectory(external/unordered_dense)
target_link_libraries(${PROJECT_NAME} PUBLIC unordered_dense::unordered_dense)

if(KMM_BUILD_TESTS)
  add_subdirectory(external/Catch2)
  add_subdirectory(test)
endif()

