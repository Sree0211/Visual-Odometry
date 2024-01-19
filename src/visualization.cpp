#include "visualization.hpp"

Visualization::Visualization(int width, int height) : windowWidth(width), windowHeight(height) {
    initializeOpenGL();
}

Visualization::~Visualization(){
    glfwTerminate();
}

void Visualization::setGTPoses(const std::vector<std::pair<double, double>>& poses) {
    this->gtPoses = poses;
}

void Visualization::setEstimatedPoses(const std::vector<std::pair<double, double>>& poses) {
    this->estimatedPoses = poses;
}

void Visualization::processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void Visualization::initializeOpenGL() {
    
    // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    if (!glfwInit()) {
    // Handle GLFW initialization error
    return;
    }

    // Create a windowed mode window and its OpenGL context
    this->window = glfwCreateWindow(this->windowWidth, this->windowHeight, "Pose Visualization", NULL, NULL);
    
    if (!this->window) {
        // Handle window creation error
        glfwTerminate();
        return;
    }

    // Make the window's context current
    glfwMakeContextCurrent(this->window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        // Handle GLEW initialization error
        return;
    }

    // Set the clear color
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

void Visualization::drawPoses()
{
    glPointSize(5.0f);

    // Draw ground truth poses in blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_POINTS);
    for (const auto& pose : this->gtPoses) {
        glVertex2d(pose.first, pose.second);
    }
    glEnd();

    // Check for OpenGL errors after the first group of commands
    GLenum err1 = glGetError();
    if (err1 != GL_NO_ERROR) {
        std::cerr << "OpenGL Error (drawPoses 1): " << err1 << std::endl;
        return;
    }

    // Draw estimated poses in red
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_POINTS);
    for (const auto& pose : this->estimatedPoses) {
        glVertex2d(pose.first, pose.second);
    }
    glEnd();

    // Check for OpenGL errors after the second group of commands
    GLenum err2 = glGetError();
    if (err2 != GL_NO_ERROR) {
        std::cerr << "OpenGL Error (drawPoses 2): " << err2 << std::endl;
        return;
    }
}


void Visualization::runVisualization()
{   
    // Make the window's context current
    glfwMakeContextCurrent(this->window);

    // Check for initialization errors
    GLenum initErr = glGetError();
    if (initErr != GL_NO_ERROR) {
        std::cerr << "OpenGL Error (Initialization): " << initErr << std::endl;
        return;
    }

    while (!glfwWindowShouldClose(this->window)) {
        glClear(GL_COLOR_BUFFER_BIT);  // Clear the previous frame

        this->processInput(window);

        // Draw ground truth and estimated poses for our Visual Odometry
        this->drawPoses();

        // Check for OpenGL errors after each draw command
        GLenum drawErr = glGetError();
        if (drawErr != GL_NO_ERROR) {
            std::cerr << "OpenGL Error (Drawing): " << drawErr << std::endl;
            break;  // Exit the loop if an error occurs during drawing
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup and close the OpenGL context
    // glfwTerminate();
}
