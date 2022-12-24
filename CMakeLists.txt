cmake_minimum_required(VERSION 3.21)
project(Mandelbrot-set-GPU LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75) # For RTX 20XX series
endif()
# glfw
option(GLFW_BUILD_TESTS off)
option(GLFW_BUILD_EXAMPLES off)
option(GLFW_BUILD_DOCS off)
add_subdirectory(libs/glfw)
list(APPEND LIBRARIES glfw ${glfw_LIBRARIES})
# glew
option(BUILD_UTILS off)
add_subdirectory(libs/glew/build/cmake)
list(APPEND LIBRARIES glew_s)
# glm
add_subdirectory(libs/glm)
list(APPEND LIBRARIES glm::glm)
# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  message("OK, you find OpemMP!")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  message(FATAL_ERROR "openmp not found!")
endif()
add_subdirectory(libs)
include_directories(
  src
  include
  libs/glm
  libs/glew/include
  libs/glfw/include
  libs/imgui    # imgui
  libs/imguizmo # imguizmo
)
list(APPEND GUI_SOURCES
  libs/imguizmo/ImGuizmo.cpp
  libs/imgui/imgui.cpp
  libs/imgui/backends/imgui_impl_glfw.cpp
  libs/imgui/backends/imgui_impl_opengl3.cpp
  libs/imgui/imgui_draw.cpp
  libs/imgui/imgui_demo.cpp
  libs/imgui/imgui_tables.cpp
  libs/imgui/imgui_widgets.cpp
)
# set source files and add executable
file(GLOB SRC_FILES src/*.cpp)
set(CMAKE_BUILD_TYPE "Release")
add_executable(main ${SRC_FILES} ${INTERNAL_SRC_FILES} ${GUI_SOURCES})
target_include_directories(main PRIVATE include)
target_compile_features(main PRIVATE cxx_std_17)
target_compile_definitions(main PRIVATE ${DEFINES})
target_link_libraries(main PRIVATE ${LIBRARIES} stb Eigen::Eigen OpenMP::OpenMP_CXX)
