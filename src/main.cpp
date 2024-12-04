#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <iostream>
#include "external/FastNoiseLite.h"
#include <render/shader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

// Define Vertex structure
struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

// Constants
const unsigned int GRID_SIZE = 100;
const float GRID_SCALE = 1.0f;
const float HEIGHT_SCALE = 10.0f;

glm::vec3 eye_center(0.0f, 50.0f, 100.0f);
glm::vec3 lookat(0.0f, 0.0f, 0.0f);
glm::vec3 up(0.0f, 1.0f, 0.0f);
float FoV = 75.0f;
float zNear = 0.1f;
float zFar = 1000.0f;

// Function prototypes
void processInput(GLFWwindow *window);
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode);
std::vector<Vertex> generateTerrain(unsigned int gridSize, float gridScale, float heightScale, std::vector<unsigned int>& indices);
GLuint setupTerrainBuffers(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices);

GLuint loadTexture(const char* path) {
    GLuint textureID;
    glGenTextures(1, &textureID);

    int width, height, nrChannels;
    unsigned char* data = stbi_load(path, &width, &height, &nrChannels, 0);
    if (data) {
        GLenum format = (nrChannels == 4) ? GL_RGBA : GL_RGB;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    } else {
        std::cerr << "Failed to load texture at path: " << path << std::endl;
        stbi_image_free(data);
    }
    return textureID;
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(1024, 768, "Terrain Rendering with Camera", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to open a GLFW window." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD." << std::endl;
        return -1;
    }

	 GLuint grassTexture = loadTexture("../src/utils/grass.jpeg");
    if (grassTexture == 0) {
        std::cerr << "Failed to load grass texture!" << std::endl;
        return -1;
    }

    GLuint shaderProgram = LoadShadersFromFile("../src/shader/terrain.vert", "../src/shader/terrain.frag");
    if (shaderProgram == 0) {
        std::cerr << "Failed to load shaders." << std::endl;
        return -1;
    }

    std::vector<unsigned int> indices;
    std::vector<Vertex> vertices = generateTerrain(GRID_SIZE, GRID_SCALE, HEIGHT_SCALE, indices);
    GLuint terrainVAO = setupTerrainBuffers(vertices, indices);

    glm::mat4 projectionMatrix = glm::perspective(glm::radians(FoV), 1024.0f / 768.0f, zNear, zFar);

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glm::mat4 viewMatrix = glm::lookAt(eye_center, lookat, up);
        glm::mat4 vp = projectionMatrix * viewMatrix;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);
        GLint vpLoc = glGetUniformLocation(shaderProgram, "vpMatrix");
		if (vpLoc == -1) {
		    std::cerr << "vpMatrix uniform not found in shader!" << std::endl;
		    return -1;
		}
		glUniformMatrix4fv(vpLoc, 1, GL_FALSE, &vp[0][0]);


        // Bind the texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, grassTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "terrainTexture"), 0);

        // Draw the terrain
        glBindVertexArray(terrainVAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

std::vector<Vertex> generateTerrain(unsigned int gridSize, float gridScale, float heightScale, std::vector<unsigned int>& indices) {
    // Function implementation from `main.cpp`
    std::vector<Vertex> vertices;

    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    noise.SetFrequency(0.05f);

    for (unsigned int z = 0; z <= gridSize; ++z) {
        for (unsigned int x = 0; x <= gridSize; ++x) {
            Vertex vertex;
            vertex.Position = glm::vec3(x * gridScale, 0.0f, z * gridScale);
            vertex.Normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex.TexCoords = glm::vec2((float)x / gridSize, (float)z / gridSize);

            float height = noise.GetNoise((float)x, (float)z);
            vertex.Position.y = height * heightScale;

            vertices.push_back(vertex);
        }
    }

    for (unsigned int z = 0; z < gridSize; ++z) {
        for (unsigned int x = 0; x < gridSize; ++x) {
            unsigned int topLeft = z * (gridSize + 1) + x;
            unsigned int topRight = topLeft + 1;
            unsigned int bottomLeft = (z + 1) * (gridSize + 1) + x;
            unsigned int bottomRight = bottomLeft + 1;

            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }

    return vertices;
}

GLuint setupTerrainBuffers(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices) {
    GLuint VAO, VBO, EBO;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));

    glBindVertexArray(0);

    return VAO;
}


void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        eye_center.z -= 1.0f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        eye_center.z += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        eye_center.x -= 1.0f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        eye_center.x += 1.0f;
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}
