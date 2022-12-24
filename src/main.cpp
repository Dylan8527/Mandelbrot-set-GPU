#include <iostream>
#include "utils.h"
#include "MandelbrotSet.h"


// void processInput(GLFWwindow *window);
// void mouse_callback(GLFWwindow *window, double x, double y);
// void scroll_callback(GLFWwindow *window, double x, double y);

const int WIDTH = 800;
const int HEIGHT = 600;

bool firstMouse = true;
float fov = 45.f;
float lastX = WIDTH / 2.f;
float lastY = HEIGHT / 2.f;

GLFWwindow *window;

int main(int argc, char *argv[])
{
    WindowGuard windowGuard(window, WIDTH, HEIGHT, "Mandelbrot set explorer on GPU");
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    // glfwSetCursorPosCallback(window, mouse_callback);
    // glfwSetScrollCallback(window, scroll_callback);

    while (!glfwWindowShouldClose(window))
    {
        
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Close the window
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
        glfwPollEvents();
        glfwSwapBuffers(window);
    }

    return 0;
}

