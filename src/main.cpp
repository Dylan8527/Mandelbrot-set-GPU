#include <iostream>
#include <cstdlib>
#include "utils.h"
#include "MandelbrotSet.h"
#include <omp.h>

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

void DrawContents(uint8_t* data);
uint8_t* GenerateRandomData(uint32_t size);

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
        
        auto data = GenerateRandomData(WIDTH * HEIGHT * 3);
        DrawContents(data);
        delete[] data;

        // Close the window
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
        glfwPollEvents();
        glfwSwapBuffers(window);
    }

    return 0;
}

void DrawContents(uint8_t* data) {
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, data);
}

uint8_t* GenerateRandomData(uint32_t size) {
    uint8_t* data = new uint8_t[size];
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
        data[i] = static_cast<uint8_t>(rand());
    return data;
}