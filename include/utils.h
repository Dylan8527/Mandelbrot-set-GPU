#ifndef __UTILS_H__
#define __UTILS_H__

#include "defines.h"

struct WindowGuard final
{
  WindowGuard(GLFWwindow *&, const int width, const int height,
              const std::string &title);
  ~WindowGuard();
};

#endif