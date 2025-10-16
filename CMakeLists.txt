cmake_minimum_required(VERSION 3.16)
project(resnet CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_executable(resnet)              # don't also list main.cpp if it's in SRC_FILES
target_sources(resnet PRIVATE ${SRC_FILES})

target_include_directories(resnet PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(resnet PRIVATE ${OpenCV_LIBS})

add_custom_command(TARGET resnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PROJECT_SOURCE_DIR}/images"
            "$<TARGET_FILE_DIR:resnet>/images"
)