cmake_minimum_required(VERSION 3.21)
project(Mandelbrot-set-GPU LANGUAGES CXX CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75) # For RTX 20XX series
endif()
find_package(OpenMP REQUIRED)

# OpenGL
add_subdirectory(opengl) # Library: OpenGLFile::OpenGLFile


# set source files and add executable
file(GLOB SRC_FILES src/*.cpp)
set(CMAKE_BUILD_TYPE "Release")
add_executable(main ${SRC_FILES})
target_include_directories(main PRIVATE include)
target_compile_features(main PRIVATE cxx_std_17)
target_link_libraries(main PRIVATE OpenGLFile::OpenGLFile)