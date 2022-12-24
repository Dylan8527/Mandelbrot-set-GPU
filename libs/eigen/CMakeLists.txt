add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(Eigen::Eigen ALIAS Eigen)