#include <iostream>
#include <cstdlib>
#include <chrono>
#include "utils.cuh"
#include "MandelbrotSet.cuh"
#include <omp.h>
// ImGUI
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <ImGuizmo.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

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
    bool show_demo_window = false;                            // Show demo window
    MandelbrotSet set(WIDTH, HEIGHT);

    double center_x = -0.748766710846959;//-0.10109636384562;//-0.77568377; //-0.748766710846959//-1.6735 //-1.7497591451303665
    double center_y =  0.123640847970064;//0.95628651080914;//0.13646737; //0.123640847970064//0.0003318 //-0.0000000036851380
    double x_start, x_fin, y_start, y_fin;

    double scale = 1.0;
    bool auto_scaling = false;
    const double ratio = WIDTH / HEIGHT;

    vec3 theta(.85, .0, .15);
    int maxiter=500;
    double ncycle=32;
    double stripe_s=0;
    double stripe_sig=0.9;
    double step_s=0;

    // Timer
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    double average_time = 0.0;
    double average_time_count = 0.0;
    enum class EAlgorithmMode : int {
        Basic,
        Serial,
        Escape,
    };
    static constexpr const char* AlgorithmStr = "Basic\0Serial\0Escape\0\0";
    int algorithm_mode = 0;

    bool save_image = false;

    void update_scale()
    {
        x_start = (x_start-center_x)*scale+center_x;
        x_fin =(x_fin-center_x)*scale+center_x;
        y_start =(y_start-center_y)* scale+center_y;
        y_fin =(y_fin-center_y)* scale+center_y;
    }

    void update()
    {
        set.update_parameter(maxiter, ncycle, stripe_s, stripe_sig, step_s);
        if (auto_scaling) {
            scale = 0.98 * scale;
        }
        x_start = center_x - 0.5 * ratio * scale;
        x_fin = center_x + 0.5 * ratio * scale;
        y_start = center_y -  0.5 * scale;
        y_fin = center_y +  0.5 * scale;
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
        start = std::chrono::system_clock::now();

        switch(algorithm_mode){
            case (int)EAlgorithmMode::Basic:
                set.basic_algorithm(x_start, x_fin, y_start, y_fin);
            break;

            case (int)EAlgorithmMode::Serial:
                set.serial_algorithm(x_start, x_fin, y_start, y_fin);
            break;

            case (int)EAlgorithmMode::Escape:
                set.update_colormap(theta);
                set.escapetime_based_algorithm(x_start, x_fin, y_start, y_fin);
            break;
        }

        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        // update_scale();
        update();
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
            ImGui::Begin("MandelbrotSet Console"); // Create a window called "Hello, world!" and append into it.

            ImGui::Checkbox("Demo Window", &show_demo_window); // Edit bools storing our window open/close state

            ImGui::ColorEdit3("Sin Colortable", (float *)&theta); // Edit 3 floats representing a color

            ImGui::Checkbox("Auto Scaling", &auto_scaling); // Edit bools storing our window open/close state

            ImGui::Combo("Algorithm", (int *)&algorithm_mode, AlgorithmStr);

            if(ImGui::TreeNode("Rendering parameter")) 
            {
                ImGui::SliderInt("maxiter", (int *)&maxiter, 100, 2000);
                ImGui::SliderFloat("ncycle", (float *)&ncycle, 0, 200);
                ImGui::SliderFloat("stripe_s", (float *)&stripe_s, 0.0f, 32.0f);
                ImGui::SliderFloat("stripe_sig", (float *)&stripe_sig, 0.0f, 1.0f);
                ImGui::SliderInt("step_s", (int *)&step_s, 0, 100);
            }

            // Save image button
            if (ImGui::Button("Save Image"))
            {
                save_image = true;
            }

            if(save_image) {
                std::string filename = "../MandelbrotSet.png";
                stbi_write_png(filename.c_str(), WIDTH, HEIGHT, 3, set.get_data(), WIDTH * 3);
                save_image = false;
            }

            average_time += elapsed_seconds.count();
            average_time_count += 1.0;

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f * average_time, 1./average_time);
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
}

int main(int argc, char *argv[])
{
    WindowGuard windowGuard(window, WIDTH, HEIGHT, "Mandelbrot set explorer on GPU");
    glfwSetScrollCallback(window, MandelbrotSetGUI::scroll_callback);
    // glfwSetCursorPosCallback(window, MandelbrotSetGUI::mouse_callback);
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

        MandelbrotSetGUI::RenderOpenGL();
        MandelbrotSetGUI::RenderMainImGui();

        glfwSwapBuffers(window);
        glfwPollEvents();
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

uint8_t *GenerateRandomData(uint32_t size)
{
    uint8_t *data = new uint8_t[size];
#pragma omp parallel for
    for (int i = 0; i < size; i++)
        data[i] = static_cast<uint8_t>(rand());
    return data;
}