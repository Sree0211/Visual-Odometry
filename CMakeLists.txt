cmake_minimum_required(VERSION 3.12)
project(visualOdo CXX)
set(CMAKE_CXX_STANDARD 17)

# Set the include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set the library directories
link_directories(${CMAKE_SOURCE_DIR}/lib)

#find Opencv
find_package(OpenCV REQUIRED)
include_directories(visualOdo ${OpenCV_INCLUDE_DIRS})

# Locate OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# Find GLFW
find_package(glfw3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})

# Find GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})


file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
add_executable(visualOdo ${CPP_SOURCES})
target_include_directories(visualOdo PRIVATE  ${OPENGL_INCLUDE_DIR})
target_link_libraries(visualOdo glfw ${GLFW_LIBRARIES} ${OPENGL_gl_LIBRARY} ${OpenCV_LIBS} GLEW::GLEW)

target_compile_options(visualOdo PRIVATE -Wall -Wno-deprecated)