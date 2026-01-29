message(STATUS "USING OpenCV_DIR='${OpenCV_DIR}'")
file(GLOB _probe "${OpenCV_DIR}/*")
message(STATUS "CONTENTS of OpenCV_DIR: ${_probe}")


cmake_minimum_required(VERSION 3.22.1)
project(sudoku_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)
endif()

# -------- Resolve OpenCV location --------
# Priority:
# 1) -DOpenCV_DIR passed from Gradle/CMake arguments
# 2) Environment variable OPENCV_ANDROID_SDK
# 3) Repo vendor at <repo_root>/third_party/OpenCV-android-sdk
# 4) Local vendor at <this cpp dir>/third_party/OpenCV-android-sdk

# If OpenCV_DIR not given, try env var
if (NOT OpenCV_DIR AND DEFINED ENV{OPENCV_ANDROID_SDK})
    set(OpenCV_DIR "$ENV{OPENCV_ANDROID_SDK}/sdk/native/jni")
endif()

# If still not set, try repo_root/third_party
if (NOT OpenCV_DIR)
    # This file usually lives at app/src/main/cpp/CMakeLists.txt
    # Go up 3 to reach repo root: <repo_root>/app/src/main/cpp -> <repo_root>
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
    set(_REPO_VENDOR "${REPO_ROOT}/third_party/OpenCV-android-sdk/sdk/native/jni")
    if (EXISTS "${_REPO_VENDOR}/OpenCVConfig.cmake")
        set(OpenCV_DIR "${_REPO_VENDOR}")
    endif()
endif()

# NEW: if still not set, try cpp/third_party (your current layout)
if (NOT OpenCV_DIR)
    set(_CPP_VENDOR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCV-android-sdk/sdk/native/jni")
    if (EXISTS "${_CPP_VENDOR}/OpenCVConfig.cmake")
        set(OpenCV_DIR "${_CPP_VENDOR}")
    endif()
endif()

# Helpful logs
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Resolved OpenCV_DIR = ${OpenCV_DIR}")

# Fail with guidance if still not found
if (NOT EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
    message(FATAL_ERROR
            "\nOpenCVConfig.cmake not found.\n"
            "Searched:\n"
            "  OpenCV_DIR=${OpenCV_DIR}\n"
            "Try one of:\n"
            "  - Pass -DOpenCV_DIR=C:/full/path/to/OpenCV-android-sdk/sdk/native/jni\n"
            "  - Set env var OPENCV_ANDROID_SDK=C:/full/path/to/OpenCV-android-sdk\n"
            "  - Vendor under repo_root/third_party or cpp/third_party as above.\n")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)

# -------- Native sources --------
set(NATIVE_SOURCES
        native-lib.cpp
        warp_tiler.cpp
)

add_library(native-lib SHARED ${NATIVE_SOURCES})

# Android system libs
find_library(log-lib         log)
find_library(jnigraphics-lib jnigraphics)

target_include_directories(native-lib PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(native-lib
        ${OpenCV_LIBS}
        ${log-lib}
        ${jnigraphics-lib}
)

if (CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_definitions(native-lib PRIVATE NDEBUG)
endif()