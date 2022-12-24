#include "utils.cuh"

WindowGuard::WindowGuard(GLFWwindow *&window, const int width, const int height,
                         const std::string &title)
{
  if (!glfwInit())
    LOG_ERR("Failed to init glfw");

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
#endif
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  // Create window.
  if (!(window =
            glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr)))
    LOG_ERR("failed to create glfw window");

  glfwMakeContextCurrent(window);
  // Enable vsync.
  glfwSwapInterval(1);
  // Init glew.
  if (glewInit() != GLEW_OK)
    LOG_ERR("failed to init glew");
  glViewport(0, 0, width, height);
}

WindowGuard::~WindowGuard() { glfwTerminate(); }

