#include <iostream>
#include <cstdlib>
#include "utils.cuh"
#include "MandelbrotSet.cuh"
#include <omp.h>
// ImGUI
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <ImGuizmo.h>

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

void DrawContents(uint8_t *data);
uint8_t *GenerateRandomData(uint32_t size);

namespace MandelbrotSetGUI
{
    bool show_demo_window = true;                            // Show demo window
    bool show_another_window = false;                        // Show another window
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f); // Background color

    MandelbrotSet set(WIDTH, HEIGHT);

    double center_x = -0.748766710846959;//-0.10109636384562;//-0.77568377; //-0.748766710846959//-1.6735 //-1.7497591451303665
    double center_y =0.123640847970064;//0.95628651080914;//0.13646737; //0.123640847970064//0.0003318 //-0.0000000036851380
    double x_start = -2.0;
    double x_fin = 1.0;
    double y_start = -1.0;
    double y_fin = 1.0;
    double scale = 1.0;

    void update_scale()
    {
        // x_start = (x_start-center_x)*scale+center_x;
        // x_fin =(x_fin-center_x)*scale+center_x;
        // y_start =(y_start-center_y)* scale+center_y;
        // y_fin =(y_fin-center_y)* scale+center_y;
        x_start = center_x - 1.0 * scale;
        x_fin = center_x + 1.0 * scale;
        y_start = center_y - 1.0 * scale;
        y_fin = center_y + 1.0 * scale;
    }

    void processInput(GLFWwindow *window)
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        {
            center_y += 0.01*scale;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        {
            center_y -= 0.01*scale;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        {
            center_x -= 0.01*scale;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        {
            center_x += 0.01*scale;
        }
    }

    void mouse_callback(GLFWwindow *window, double x, double y)
    {
        if (firstMouse)
        {
            lastX = x;
            lastY = y;
            firstMouse = false;
        }

        float xoffset = x - lastX;
        float yoffset = lastY - y; // reversed since y-coordinates go from bottom to top

        lastX = x;
        lastY = y;

        center_x += xoffset * 0.001 * scale;
        center_y += yoffset * 0.001 * scale;
    }

    void scroll_callback(GLFWwindow* window, double x, double y)
    {
        if (y > 0)
        {
            scale *= 0.9;
        }
        else
        {
            scale *= 1.1;
        }
    }

    //-------------------------opengl drawing-------------------------------------
    void RenderOpenGL()
    {
        /*auto data = GenerateRandomData(WIDTH * HEIGHT * 3);
        DrawContents(data);
        delete[] data;*/
        set.compute(x_start, x_fin, y_start, y_fin);
        // update_scale();
        DrawContents(set.get_data());
    }

    //-------------------------imgui creation-------------------------------------
    void RenderMainImGui()
    {

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window//
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");          // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window); // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);             // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float *)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Another Window", &show_another_window); // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        // glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        // glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
}

int main(int argc, char *argv[])
{
    WindowGuard windowGuard(window, WIDTH, HEIGHT, "Mandelbrot set explorer on GPU");
    glfwSetScrollCallback(window, MandelbrotSetGUI::scroll_callback);
    glfwSetCursorPosCallback(window, MandelbrotSetGUI::mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext(); // Setup Dear ImGui context
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

    ImGui::StyleColorsDark(); // Setup Dear ImGui style
    const char *glsl_version = "#version 130";
    ImGui_ImplGlfw_InitForOpenGL(window, true); // Setup Platform/Renderer bindings
    ImGui_ImplOpenGL3_Init(glsl_version);

    while (!glfwWindowShouldClose(window))
    {
        MandelbrotSetGUI::processInput(window);
        glfwPollEvents();

        MandelbrotSetGUI::RenderOpenGL();
        MandelbrotSetGUI::RenderMainImGui();

        // Close the window
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        glfwSwapBuffers(window);
        // glfwPollEvents();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

void DrawContents(uint8_t *data)
{
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, data);
}
W
uint8_t *GenerateRandomData(uint32_t size)
{
    uint8_t *data = new uint8_t[size];
#pragma omp parallel for
    for (int i = 0; i < size; i++)
        data[i] = static_cast<uint8_t>(rand());
    return data;
}