# glfw
option(GLFW_BUILD_TESTS off)
option(GLFW_BUILD_EXAMPLES off)
option(GLFW_BUILD_DOCS off)
add_subdirectory(libs/glfw)
list(APPEND LIBARAIES glfw ${GLFW_LIBRARIES})

# glew
option(BUILD_UTILS off)
add_subdirectory(libs/glew/build/cmake)
list(APPEND LIBARAIES glew_s)

# glm
add_subdirectory(libs/glm)
list(APPEND LIBARAIES glm::glm)

# eigen & stb
add_subdirectory(libs)

include_directories(
    include
    libs/glm
    libs/glew/include
    libs/glfw/include
)
file(GLOB SRC_FILES "*.cpp") # OpenGL on CPU/Host
add_library(OpenGLFile STATIC ${SRC_FILES} ${INTERNAL_SRC_FILES})
target_link_libraries(OpenGLFile PUBLIC ${LIBARAIES} stb Eigen::Eigen)
target_compile_features(OpenGLFile PUBLIC cxx_std_17)
target_include_directories(OpenGLFile PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(OpenGLFile::OpenGLFile ALIAS OpenGLFile)