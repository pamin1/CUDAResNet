cmake_minimum_required(VERSION 3.16)

# Enable CUDA as a language (for now or future .cu files)
project(resnet LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

# This finds the CUDA toolkit and gives you imported targets like CUDA::cudart
find_package(CUDAToolkit REQUIRED)

# Grab your sources (.cpp and any .cu)
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.cu"
)

add_executable(resnet)

add_subdirectory("include/cnpy")

target_sources(resnet PRIVATE ${SRC_FILES})

target_include_directories(resnet PRIVATE "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(resnet PRIVATE
    cnpy
    nlohmann_json::nlohmann_json
    ZLIB::ZLIB
    ${OpenCV_LIBS}
    CUDA::cudart         # <-- THIS is the important line
)

# (optional but nice if/when you add real kernels)
set_target_properties(resnet PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_custom_command(TARGET resnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PROJECT_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:resnet>/assets"
)
