cmake_minimum_required(VERSION 3.16)
project(resnet CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)  # provides nlohmann_json::nlohmann_json
find_package(ZLIB REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_executable(resnet)
add_subdirectory("include/cnpy")
# add_subdirectory("include/json")

target_sources(resnet PRIVATE ${SRC_FILES})

target_include_directories(resnet PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(resnet PRIVATE cnpy nlohmann_json::nlohmann_json ZLIB::ZLIB ${OpenCV_LIBS})

add_custom_command(TARGET resnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PROJECT_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:resnet>/assets"
)