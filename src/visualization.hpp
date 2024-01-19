#ifndef VISUALIZATION_H
#define VISUALIZATION_H

// #include <glad/glad.h>
#include <GL/glew.h>
#include <OpenGL/gl.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <iostream>


class Visualization
{

    public:
    Visualization(int width, int height);
    ~Visualization();

    void setEstimatedPoses(const std::vector<std::pair<double, double>>& poses);
    void setGTPoses(const std::vector<std::pair<double, double>>& poses);
    

    void runVisualization();
    

    private:

    GLFWwindow* window;
    int windowWidth;
    int windowHeight;

    std::vector<std::pair<double, double>> gtPoses;
    std::vector<std::pair<double, double>> estimatedPoses;

    void initializeOpenGL();
    void processInput(GLFWwindow *window);
    void drawPoses();
};

#endif