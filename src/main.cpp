#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include "external/FastNoiseLite.h"
#include <render/shader.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#include <tinygltf-2.9.3/tiny_gltf.h>

// Define Vertex structure
struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

struct TurbineMesh {
    GLuint VAO;
    GLuint EBO;
    GLsizei indexCount;
    GLenum indexType; // GL_UNSIGNED_SHORT or GL_UNSIGNED_INT depending on the accessor
    GLsizei vertexCount;
};

struct Turbine {
    tinygltf::Model model;
    std::vector<TurbineMesh> meshes;
};

// Constants
const unsigned int GRID_SIZE = 100;
const float GRID_SCALE = 1.0f;
const float HEIGHT_SCALE = 10.0f;

glm::vec3 eye_center(50.0f, 100.0f, -150.0f);
glm::vec3 lookat(50.0f, 50.0f, 50.0f);
glm::vec3 up(0.0f, 1.0f, 0.0f);
glm::vec3 sunlightDirection = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f)); 
glm::vec3 sunlightColor = glm::vec3(1.0f, 1.0f, 0.9f);                        
float FoV = 75.0f;
float zNear = 0.1f;
float zFar = 5000.0f;
glm::vec3 forwardDirection = glm::normalize(lookat - eye_center);
glm::vec3 rightDirection   = glm::normalize(glm::cross(forwardDirection, up));
float cameraViewDistance = glm::length(lookat - eye_center);
std::vector<glm::mat4> turbineInstances;
GLuint instanceVBO;
const int NUM_TURBINES = 20;

// Global chunk coordinates
int currentChunkX = 0;
int currentChunkZ = 0;

// Function prototypes
void processInput(GLFWwindow *window);
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode);
std::vector<Vertex> generateTerrain(unsigned int gridSize, float gridScale, float heightScale, std::vector<unsigned int>& indices, int chunkX, int chunkZ);
GLuint setupTerrainBuffers(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices);
void updateChunks(int chunkX, int chunkZ);
void renderTerrainChunks(GLuint shader, const glm::mat4& vpMatrix, GLuint texture);
void renderSun(GLuint shader, GLuint sunVAO, const glm::mat4& vpMatrix);
void renderTurbine(const Turbine& turbine, GLuint shader, const glm::mat4& vpMatrix);
void generateTurbineInstances();

Turbine loadTurbine(const char* path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool success = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    if (!warn.empty()) {
        std::cerr << "Warning: " << warn << std::endl;
    }
    if (!success) {
        std::cerr << "Failed to load turbine model: " << err << std::endl;
        return {}; 
    }

    std::vector<TurbineMesh> turbineMeshes;

    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            GLuint vao;
            glGenVertexArrays(1, &vao);
            glBindVertexArray(vao);

            // Setup attributes
            for (auto& attrib : primitive.attributes) {
                const auto& accessor = model.accessors[attrib.second];
                const auto& bufferView = model.bufferViews[accessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];

                GLuint vbo;
                glGenBuffers(1, &vbo);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, bufferView.byteLength,
                             &buffer.data[bufferView.byteOffset], GL_STATIC_DRAW);

                GLint attribLocation = -1;
                if (attrib.first == "POSITION") attribLocation = 0;
                if (attrib.first == "NORMAL") attribLocation = 1;
                if (attrib.first == "TEXCOORD_0") attribLocation = 2;

                if (attribLocation >= 0) {
                    int componentCount = 0;
                    switch (accessor.type) {
                        case TINYGLTF_TYPE_VEC2: componentCount = 2; break;
                        case TINYGLTF_TYPE_VEC3: componentCount = 3; break;
                        case TINYGLTF_TYPE_VEC4: componentCount = 4; break;
                        case TINYGLTF_TYPE_SCALAR: componentCount = 1; break;
                        default: break;
                    }

                    glEnableVertexAttribArray(attribLocation);
                    glVertexAttribPointer(
                        attribLocation,
                        componentCount,
                        accessor.componentType,
                        accessor.normalized ? GL_TRUE : GL_FALSE,
                        int(accessor.ByteStride(bufferView)),
                        (void*)(uintptr_t)accessor.byteOffset
                    );
                }
            }

            // Prepare the TurbineMesh struct
            TurbineMesh tmesh;
            tmesh.VAO = vao;
            tmesh.EBO = 0;
            tmesh.indexCount = 0;
            tmesh.indexType = GL_UNSIGNED_INT;
            tmesh.vertexCount = 0;

            if (primitive.indices >= 0) {
                // Indexed geometry setup
                const auto& indexAccessor = model.accessors[primitive.indices];
                const auto& bufferView = model.bufferViews[indexAccessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];

                GLuint ebo;
                glGenBuffers(1, &ebo);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, bufferView.byteLength, &buffer.data[bufferView.byteOffset], GL_STATIC_DRAW);

                GLsizei indexCount = static_cast<GLsizei>(indexAccessor.count);
                GLenum indexType = GL_UNSIGNED_INT;

                if (indexAccessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
                    indexType = GL_UNSIGNED_SHORT;
                } else if (indexAccessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT) {
                    indexType = GL_UNSIGNED_INT;
                } else if (indexAccessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE) {
                    indexType = GL_UNSIGNED_BYTE;
                }

                tmesh.EBO = ebo;
                tmesh.indexCount = indexCount;
                tmesh.indexType = indexType;
            } else {
                // Non-indexed geometry
                int positionIndex = primitive.attributes.at("POSITION");
                const auto& positionAccessor = model.accessors[positionIndex];
                GLsizei vertexCount = static_cast<GLsizei>(positionAccessor.count);
                tmesh.vertexCount = vertexCount;
            }

            turbineMeshes.push_back(tmesh);

            glBindVertexArray(0);
        }
    }

    Turbine turbine;
    turbine.model = model;
    turbine.meshes = turbineMeshes;
    return turbine;
}


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

struct Chunk {
    GLuint VAO;
    unsigned int indexCount;
    glm::vec2 position;  
};

std::vector<Chunk> activeChunks;

void generateSphere(float radius, int sectorCount, int stackCount, 
                    std::vector<float>& vertexData, 
                    std::vector<unsigned int>& indices) {
    // vertexData will store: position.x, position.y, position.z, normal.x, normal.y, normal.z
    const float PI = 3.14159265359f;

    for (int i = 0; i <= stackCount; ++i) {
        float stackAngle = PI/2 - i * (PI/stackCount); 
        float xy = radius * cosf(stackAngle);   
        float z = radius * sinf(stackAngle);    

        for (int j = 0; j <= sectorCount; ++j) {
            float sectorAngle = j * (2 * PI / sectorCount);
            float x = xy * cosf(sectorAngle);
            float y = xy * sinf(sectorAngle);

            // Position
            vertexData.push_back(x);
            vertexData.push_back(y);
            vertexData.push_back(z);

            // Normal (normalize the vector since it's a sphere)
            glm::vec3 normal = glm::normalize(glm::vec3(x, y, z));
            vertexData.push_back(normal.x);
            vertexData.push_back(normal.y);
            vertexData.push_back(normal.z);
        }
    }

    // Indices
    for (int i = 0; i < stackCount; ++i) {
        int k1 = i * (sectorCount + 1);
        int k2 = k1 + sectorCount + 1;

        for (int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
            indices.push_back(k1);
            indices.push_back(k2);
            indices.push_back(k1 + 1);

            indices.push_back(k1 + 1);
            indices.push_back(k2);
            indices.push_back(k2 + 1);
        }
    }
}


GLuint createSunVAO() {
    std::vector<float> vertexData;
    std::vector<unsigned int> indices;

    int sectorCount = 36;
    int stackCount  = 18;
    generateSphere(1.0f, sectorCount, stackCount, vertexData, indices);

    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // Each vertex has 6 floats: position(3) + normal(3)
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Position attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);

    // Normal attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);

    return VAO;
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

    GLFWwindow *window = glfwCreateWindow(1024, 768, "Infinite Terrain", NULL, NULL);
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

    GLuint terrainShader = LoadShadersFromFile("../src/shader/terrain.vert", "../src/shader/terrain.frag");
    if (terrainShader == 0) {
        std::cerr << "Failed to load terrain shaders." << std::endl;
        return -1;
    }

    GLuint sunLightingShader = LoadShadersFromFile("../src/shader/sun.vert", "../src/shader/sun.frag");
    if (sunLightingShader == 0) {
        std::cerr << "Failed to load sun lighting shaders." << std::endl;
        return -1;
    }

    GLuint turbineShader = LoadShadersFromFile("../src/shader/turbine.vert", "../src/shader/turbine.frag");
    if (turbineShader == 0) {
        std::cerr << "Failed to load turbine shaders." << std::endl;
        return -1;
    }

    GLuint sunVAO = createSunVAO();

    updateChunks(currentChunkX, currentChunkZ);

    glm::mat4 projectionMatrix = glm::perspective(glm::radians(FoV), 1024.0f / 768.0f, zNear, zFar);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.5f, 0.7f, 1.0f, 1.0f); 

    Turbine turbine = loadTurbine("../src/model/turbine/Turbine.glb");

    generateTurbineInstances();
    
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, turbineInstances.size() * sizeof(glm::mat4), &turbineInstances[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    for (auto& tmesh : turbine.meshes) {
        glBindVertexArray(tmesh.VAO);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    
        std::size_t vec4Size = sizeof(glm::vec4);
        for (int i = 0; i < 4; i++) {
            glEnableVertexAttribArray(3 + i);
            glVertexAttribPointer(3 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(i * vec4Size));
            glVertexAttribDivisor(3 + i, 1);
        }
    
        glBindVertexArray(0);
    }
    
    while (!glfwWindowShouldClose(window)) {
        processInput(window);  
    
        glm::mat4 viewMatrix = glm::lookAt(eye_center, lookat, up);
        glm::mat4 vpMatrix = projectionMatrix * viewMatrix;
    
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
        renderTerrainChunks(terrainShader, vpMatrix, grassTexture);
        renderSun(sunLightingShader, sunVAO, vpMatrix);
        renderTurbine(turbine, turbineShader, vpMatrix);
    
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void renderTerrainChunks(GLuint shader, const glm::mat4& vpMatrix, GLuint texture) {
    glUseProgram(shader);

    glUniformMatrix4fv(glGetUniformLocation(shader, "vpMatrix"), 1, GL_FALSE, &vpMatrix[0][0]);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glUniform1i(glGetUniformLocation(shader, "terrainTexture"), 0);

        for (const auto& chunk : activeChunks) {
            GLint chunkOffsetLoc = glGetUniformLocation(shader, "chunkOffset");
            if (chunkOffsetLoc == -1) {
                std::cerr << "chunkOffset uniform not found in shader!" << std::endl;
                continue;
            }
            glUniform2f(chunkOffsetLoc, chunk.position.x, chunk.position.y);

            glBindVertexArray(chunk.VAO);
            glDrawElements(GL_TRIANGLES, chunk.indexCount, GL_UNSIGNED_INT, 0);
        }
}

void renderSun(GLuint shader, GLuint sunVAO, const glm::mat4& vpMatrix) {
    glUseProgram(shader);

    float forwardDistance = 200.0f; 
    float rightOffset     = 125.0f; 
    float upOffset        = 100.0f; 
    glm::vec3 sunPosition = eye_center + forwardDirection * forwardDistance + rightDirection * rightOffset + up * upOffset;

    glm::mat4 model = glm::translate(glm::mat4(1.0f), sunPosition);
    model = glm::scale(model, glm::vec3(10.0f));

    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &model[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shader, "vpMatrix"), 1, GL_FALSE, &vpMatrix[0][0]);

    // Light uniforms
    glUniform3fv(glGetUniformLocation(shader, "lightColor"), 1, &sunlightColor[0]);
    glUniform1f(glGetUniformLocation(shader, "intensity"), 1.0f);

    // Example directional light direction
    glm::vec3 dir = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));
    glUniform3fv(glGetUniformLocation(shader, "lightDir"), 1, &dir[0]);

    // Draw the sphere
    glBindVertexArray(sunVAO);
    // Sphere has (sectorCount * stackCount * 6) indices
    glDrawElements(GL_TRIANGLES, 36 * 18 * 6, GL_UNSIGNED_INT, 0);
}

void renderTurbine(const Turbine& turbine, GLuint shader, const glm::mat4& vpMatrix) {
    glUseProgram(shader);

    static double lastTime = glfwGetTime();
    double currentTime = glfwGetTime();
    float deltaTime = float(currentTime - lastTime);
    lastTime = currentTime;

    static float bladeRotation = 0.0f;
    float rotationSpeed = 20.0f;
    bladeRotation += deltaTime * rotationSpeed;
    bladeRotation = fmod(bladeRotation, 360.0f);

    glm::mat4 bladeRotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(bladeRotation), glm::vec3(0.0f, 0.0f, 1.0f));
    
    glUniformMatrix4fv(glGetUniformLocation(shader, "vpMatrix"), 1, GL_FALSE, &vpMatrix[0][0]);
    glUniform3f(glGetUniformLocation(shader, "lightColor"), 1.0f, 1.0f, 1.0f);
    glUniform3f(glGetUniformLocation(shader, "lightDir"), -1.0f, -1.0f, -1.0f);
    glUniform3f(glGetUniformLocation(shader, "viewPos"), eye_center.x, eye_center.y, eye_center.z);

    for (size_t i = 0; i < turbine.meshes.size(); ++i) {
        bool isBladeMesh = (i == 16);

        glUniform1i(glGetUniformLocation(shader, "isBlade"), isBladeMesh ? 1 : 0);
        glUniformMatrix4fv(glGetUniformLocation(shader, "bladeRotationMatrix"), 1, GL_FALSE, &bladeRotationMatrix[0][0]);

        glBindVertexArray(turbine.meshes[i].VAO);

        if (turbine.meshes[i].indexCount > 0) {
            glDrawElementsInstanced(GL_TRIANGLES, turbine.meshes[i].indexCount, turbine.meshes[i].indexType, 0, NUM_TURBINES);
        } else {
            glDrawArraysInstanced(GL_TRIANGLES, 0, turbine.meshes[i].vertexCount, NUM_TURBINES);
        }
    }
}

void generateTurbineInstances() {
    turbineInstances.clear();
    turbineInstances.reserve(NUM_TURBINES);

    srand(42);
    for (int i = 0; i < NUM_TURBINES; i++) {
        float x = (rand() % 1000) - 500.0f;
        float z = (rand() % 1000) - 500.0f;
        float y = 0.0f; // You could sample the terrain height if needed

        float angle = glm::radians((float)(rand() % 360));

        glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
        model = glm::rotate(model, angle, glm::vec3(0,1,0));
        turbineInstances.push_back(model);
    }
}

void updateChunks(int chunkX, int chunkZ) {
    activeChunks.clear(); 
    int range = 5;

    for (int dz = -range; dz <= range; ++dz) {
        for (int dx = -range; dx <= range; ++dx) {
            int cx = chunkX + dx;
            int cz = chunkZ + dz;
            
            // Just store chunk coordinates; no scaling here in the CPU vertices
            glm::vec2 chunkPos(cx * GRID_SIZE * GRID_SCALE, cz * GRID_SIZE * GRID_SCALE);

            std::vector<unsigned int> indices;
            std::vector<Vertex> vertices = generateTerrain(GRID_SIZE, GRID_SCALE, HEIGHT_SCALE, indices, cx, cz);

            GLuint terrainVAO = setupTerrainBuffers(vertices, indices);
            activeChunks.push_back({terrainVAO, (unsigned int)indices.size(), chunkPos});
        }
    }
}

std::vector<Vertex> generateTerrain(unsigned int gridSize, float gridScale, float heightScale, 
                                    std::vector<unsigned int>& indices, int chunkX, int chunkZ)
{
    std::vector<Vertex> vertices;
    vertices.reserve((gridSize + 1) * (gridSize + 1));
    
    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    noise.SetFrequency(0.05f);

    float worldOffsetX = chunkX * (float)gridSize * gridScale;
    float worldOffsetZ = chunkZ * (float)gridSize * gridScale;

    for (unsigned int z = 0; z <= gridSize; ++z) {
        for (unsigned int x = 0; x <= gridSize; ++x) {
            float localX = x * gridScale;
            float localZ = z * gridScale;

            float globalX = worldOffsetX + localX;
            float globalZ = worldOffsetZ + localZ;

            float height = noise.GetNoise(globalX, globalZ) * heightScale;

            Vertex vertex;
            vertex.Position = glm::vec3(localX, height, localZ);
            vertex.Normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex.TexCoords = glm::vec2((float)x / gridSize, (float)z / gridSize);

            vertices.push_back(vertex);
        }
    }

    indices.reserve(gridSize * gridSize * 6);
    for (unsigned int z = 0; z < gridSize; ++z) {
        for (unsigned int x = 0; x < gridSize; ++x) {
            unsigned int topLeft = z * (gridSize + 1) + x;
            unsigned int topRight = topLeft + 1;
            unsigned int bottomLeft = (z + 1)*(gridSize + 1) + x;
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
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

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
    static float movementSpeed = 1.0f;
    glm::vec3 movement(0.0f);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        movement.z -= movementSpeed;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        movement.z += movementSpeed;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        movement.x -= movementSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        movement.x += movementSpeed;

    eye_center += movement;

    float chunkSize = GRID_SIZE * GRID_SCALE;

    int newChunkX = static_cast<int>(std::floor(eye_center.x / chunkSize));
    int newChunkZ = static_cast<int>(std::floor(eye_center.z / chunkSize));

    if (newChunkX != currentChunkX || newChunkZ != currentChunkZ) {
        currentChunkX = newChunkX;
        currentChunkZ = newChunkZ;
        updateChunks(currentChunkX, currentChunkZ);
    }

    lookat = eye_center + forwardDirection * cameraViewDistance;
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}
