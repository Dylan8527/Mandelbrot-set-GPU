#include <iostream>
#include <cstdlib>
#include "utils.h"
#include "MandelbrotSet.h"
#include <omp.h>
// ImGUI 
#  include <imgui.h>
#  include <backends/imgui_impl_glfw.h>
#  include <backends/imgui_impl_opengl3.h>
#  include <ImGuizmo.h>

// void processInput(GLFWwindow *window);
// void mouse_callback(GLFWwindow *window, double x, double y);
// void scroll_callback(GLFWwindow *window, double x, double y);

const int WIDTH = 1280;
const int HEIGHT = 720;

bool firstMouse = true;
float fov = 45.f;
float lastX = WIDTH / 2.f;
float lastY = HEIGHT / 2.f;

GLFWwindow *window;

void DrawContents(uint8_t* data);
uint8_t* GenerateRandomData(uint32_t size);

namespace MandelbrotSetGUI {
    bool show_demo_window = true; // Show demo window
    bool show_another_window = false; // Show another window
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f); // Background color

    //-------------------------opengl drawing-------------------------------------
    void RenderOpenGL()
    {
        auto data = GenerateRandomData(WIDTH * HEIGHT * 3);
        DrawContents(data);
        delete[] data;
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

            ImGui::Text("This is some useful text."); // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window); // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float *)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if(show_another_window)
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
    // glfwSetScrollCallback(window, scroll_callback);
    // glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext(); // Setup Dear ImGui context
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    ImGui::StyleColorsDark(); // Setup Dear ImGui style
    const char* glsl_version = "#version 130";
    ImGui_ImplGlfw_InitForOpenGL(window, true); // Setup Platform/Renderer bindings
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        MandelbrotSetGUI::RenderOpenGL();
        MandelbrotSetGUI::RenderMainImGui();

        // Close the window
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
        
        glfwSwapBuffers(window);
        // glfwPollEvents();
    }

    //Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

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