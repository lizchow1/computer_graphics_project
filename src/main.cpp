#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <external/FastNoiseLite.h>
#include <render/shader.h>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#include <tinygltf-2.9.3/tiny_gltf.h>

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

struct TurbineMesh {
    GLuint VAO;
    GLuint EBO;
    GLsizei indexCount;
    GLenum indexType; 
    GLsizei vertexCount;
};

struct Turbine {
    tinygltf::Model model;
    std::vector<TurbineMesh> meshes;
};

struct SolarPanelMesh {
    GLuint VAO;
    GLuint EBO;
    GLsizei indexCount;
    GLenum indexType;
    GLsizei vertexCount;
};

struct SolarPanel {
    tinygltf::Model model;
    std::vector<SolarPanelMesh> meshes;
};

struct LODLevel {
    GLuint VAO;
    unsigned int indexCount;
};

struct Chunk {
    std::vector<LODLevel> lodLevels;
    glm::vec2 position;
    int chunkX;
    int chunkZ;
};

struct ChunkData {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    glm::vec2 position;
    int chunkX;
    int chunkZ;
};

// Constants
const unsigned int GRID_SIZE = 100;
const float GRID_SCALE = 1.0f;
const float HEIGHT_SCALE = 50.0f;
const int NUM_TURBINES = 20;

glm::vec3 eye_center(0.0f, 50.0f, 2000.0f);  
glm::vec3 lookat(750.0f, 0.0f, 751.0f);       
glm::vec3 up(0.0f, 1.0f, 0.0f); 
glm::vec3 sunlightDirection = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f)); 
glm::vec3 sunlightColor = glm::vec3(1.0f, 0.9f, 0.7f);                   
glm::vec3 forwardDirection = glm::normalize(lookat - eye_center);
glm::vec3 rightDirection   = glm::normalize(glm::cross(forwardDirection, up));    
float FoV = 45.0f; 
float zNear = 0.1f;
float zFar = 3000.0f;
float cameraViewDistance = 50.0f;
float getTerrainHeight(float globalX, float globalZ);
std::vector<glm::mat4> turbineInstances;
std::vector<Chunk> activeChunks;
std::vector<Vertex> generateTerrain(unsigned int gridSize, float gridScale, float heightScale, std::vector<unsigned int>& indices, int chunkX, int chunkZ);
std::vector<glm::mat4> solarPanelInstances;
GLuint setupTerrainBuffers(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices);
GLuint instanceVBO;
GLuint solarPanelInstanceVBO;
GLuint baseColor, normalMap, metallicMap, roughnessMap;
GLuint aoMap, heightMap, emissiveMap, opacityMap, specularMap;
static float lastFrameTime = 0.0f; 
static float deltaTime = 0.0f;
int currentChunkX = 0;
int currentChunkZ = 0;
static std::thread chunkThread;
static std::mutex chunkMutex;
static std::queue<std::vector<ChunkData>> chunkDataQueue;  
static std::queue<std::pair<int,int>> chunkRequests;  
static std::atomic<bool> keepLoadingChunks(true);
static double lastTime = 0.0;
static int nbFrames = 0;

// Function prototypes
void processInput(GLFWwindow *window, float deltaTime);
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode);
void updateChunks(int chunkX, int chunkZ);
void renderTerrainChunks(GLuint shader, const glm::mat4& vpMatrix, GLuint texture, glm::mat4 lightSpaceMatrix, GLuint depthMap);
void renderSun(GLuint shader, GLuint sunVAO, const glm::mat4& vpMatrix);
void renderTurbine(const Turbine& turbine, GLuint shader, const glm::mat4& vpMatrix, glm::mat4 lightSpaceMatrix, GLuint depthMap);
void generateTurbineInstances();
void generateSolarPanelInstances(int panelCount);
void renderSolarPanels(const SolarPanel& solarPanel, GLuint shader, const glm::mat4& vpMatrix,
                       GLuint baseColor, GLuint normalMap, GLuint metallicMap, GLuint roughnessMap,
                       GLuint aoMap, GLuint heightMap, GLuint emissiveMap, GLuint opacityMap, GLuint specularMap, glm::mat4 lightSpaceMatrix, GLuint depthMap);
void renderHalo(GLuint shader, GLuint haloQuadVAO, const glm::mat4& vpMatrix);
void chunkLoadingTask();
int getLODIndex(float distance);

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
                if (attrib.first == "TEXCOORD_1") {
                    std::cerr << "Warning: TEXCOORD_1 is present but ignored." << std::endl;
                    continue; // Skip TEXCOORD_1
                }

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

SolarPanel loadSolarPanel(const char* path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    // Load the GLTF model from file
    bool success = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    if (!warn.empty()) {
        std::cerr << "Warning: " << warn << std::endl;
    }
    if (!success) {
        std::cerr << "Failed to load solar panel model: " << err << std::endl;
        return {};
    }

    std::vector<SolarPanelMesh> solarPanelMeshes;

    // Iterate over the meshes in the GLTF model
    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            GLuint vao;
            glGenVertexArrays(1, &vao);
            glBindVertexArray(vao);

            // Bind the vertex attributes
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
                if (attrib.first == "TEXCOORD_1") attribLocation = 3;

                if (attribLocation >= 0) {
                    int componentCount = 0;
                    switch (accessor.type) {
                        case TINYGLTF_TYPE_VEC2: componentCount = 2; break;
                        case TINYGLTF_TYPE_VEC3: componentCount = 3; break;
                        case TINYGLTF_TYPE_VEC4: componentCount = 4; break;
                        default: 
                            std::cerr << "Unsupported accessor type for attribute " << attrib.first << std::endl;
                            continue;
                    }

                    glEnableVertexAttribArray(attribLocation);
                    glVertexAttribPointer(
                        attribLocation,
                        componentCount,
                        accessor.componentType,
                        accessor.normalized ? GL_TRUE : GL_FALSE,
                        accessor.ByteStride(bufferView) > 0 ? accessor.ByteStride(bufferView) : componentCount * sizeof(float),
                        (void*)(uintptr_t)accessor.byteOffset
                    );
                } else {
                    std::cerr << "Unrecognized attribute: " << attrib.first << std::endl;
                }
            }

            // Prepare the SolarPanelMesh struct
            SolarPanelMesh spMesh = {vao, 0, 0, GL_UNSIGNED_INT, 0};

            // Handle indices for indexed geometry
            if (primitive.indices >= 0) {
                const auto& indexAccessor = model.accessors[primitive.indices];
                const auto& bufferView = model.bufferViews[indexAccessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];

                GLuint ebo;
                glGenBuffers(1, &ebo);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, bufferView.byteLength,
                             &buffer.data[bufferView.byteOffset], GL_STATIC_DRAW);

                spMesh.EBO = ebo;
                spMesh.indexCount = indexAccessor.count;

                // Determine index type
                switch (indexAccessor.componentType) {
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                        spMesh.indexType = GL_UNSIGNED_BYTE;
                        break;
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
                        spMesh.indexType = GL_UNSIGNED_SHORT;
                        break;
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
                        spMesh.indexType = GL_UNSIGNED_INT;
                        break;
                    default:
                        std::cerr << "Unsupported index component type in GLTF file." << std::endl;
                        break;
                }
            } else {
                // Non-indexed geometry
                const auto& positionAccessor = model.accessors[primitive.attributes.at("POSITION")];
                spMesh.vertexCount = positionAccessor.count;
            }

            solarPanelMeshes.push_back(spMesh);

            glBindVertexArray(0); // Unbind the VAO
        }
    }

    SolarPanel solarPanel;
    solarPanel.model = model;
    solarPanel.meshes = solarPanelMeshes;

    return solarPanel;
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

GLuint createSkyQuadVAO()
{
    // Two triangles to cover the full screen in clip space coordinates:
    float skyVertices[] = {
        //   X      Y
        -1.0f, -1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f,
        -1.0f,  1.0f
    };

    unsigned int skyIndices[] = { 0, 1, 2,  2, 3, 0 };

    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    // Vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyVertices), skyVertices, GL_STATIC_DRAW);

    // Index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(skyIndices), skyIndices, GL_STATIC_DRAW);

    // Positions (location = 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);

    glBindVertexArray(0);
    return VAO;
}

void generateSphere(float radius, int sectorCount, int stackCount, 
                    std::vector<float>& vertexData, 
                    std::vector<unsigned int>& indices) {
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

GLuint createHaloQuadVAO()
{
    // 4 vertices (2 triangles). Positions + UVs
    float vertices[] = {
        //   X      Y     Z     U     V
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f
    };

    unsigned int indices[] = { 0,1,2,  2,3,0 };

    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    // Vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Positions (location = 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);

    // UVs (location = 1)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3*sizeof(float)));

    glBindVertexArray(0);

    return VAO;
}

void pollLoadedChunks()
{
    std::lock_guard<std::mutex> lock(chunkMutex);
    while (!chunkDataQueue.empty())
    {
        // We now expect chunkDataQueue.front() to contain *multiple* LOD data
        std::vector<ChunkData> lodChunkData = chunkDataQueue.front();
        chunkDataQueue.pop();

        // The first LODChunkData item determines the position and chunk coords
        glm::vec2 pos   = lodChunkData[0].position;
        int       cX    = lodChunkData[0].chunkX;
        int       cZ    = lodChunkData[0].chunkZ;

        Chunk newChunk;
        newChunk.position = pos;
        newChunk.chunkX   = cX;
        newChunk.chunkZ   = cZ;

        // Build VAOs for each LOD level
        for (auto& cd : lodChunkData)
        {
            GLuint vao = setupTerrainBuffers(cd.vertices, cd.indices);
            LODLevel level;
            level.VAO        = vao;
            level.indexCount = static_cast<unsigned int>(cd.indices.size());
            newChunk.lodLevels.push_back(level);
        }

        activeChunks.push_back(newChunk);
    }
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

    GLFWwindow *window = glfwCreateWindow(1024, 768, "Towards a Futuristic Emerald Isle", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to open a GLFW window." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    lastTime = glfwGetTime();
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD." << std::endl;
        return -1;
    }

    const unsigned int SHADOW_WIDTH = 2048, SHADOW_HEIGHT = 2048;

    GLuint depthMapFBO;
    glGenFramebuffers(1, &depthMapFBO);

    // Create the depth texture
    GLuint depthMap;
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0,  GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Prevent shadow edges from clamping incorrectly
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    // Attach depth texture to FBO
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glDrawBuffer(GL_NONE); 
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    GLuint grassTexture = loadTexture("../src/utils/grass.jpeg");
    if (grassTexture == 0) {
        std::cerr << "Failed to load grass texture!" << std::endl;
        return -1;
    }

    GLuint baseColor = loadTexture("../src/utils/Solar Panel_Solar panel_BaseColor_4.png");
    if (baseColor == 0) {
        std::cerr << "Failed to load Base Color texture!" << std::endl;
        return -1;
    }

    GLuint normalMap = loadTexture("../src/utils/Solar Panel_Solar panel_Normal_3.png");
    if (normalMap == 0) {
        std::cerr << "Failed to load Normal Map texture!" << std::endl;
        return -1;
    }

    GLuint metallicMap = loadTexture("../src/utils/Solar_panel_stand_Solar_Material.001_Metallic-Solar_panel_st.png");
    if (metallicMap == 0) {
        std::cerr << "Failed to load Metallic Map texture!" << std::endl;
        return -1;
    }

    GLuint roughnessMap = loadTexture("../src/utils/Solar_panel_stand_Solar_Material.001_Normal_0.png");
    if (roughnessMap == 0) {
        std::cerr << "Failed to load Roughness Map texture!" << std::endl;
        return -1;
    }

    GLuint aoMap = loadTexture("../src/utils/Solar_panel_stand_Solar_Material.001_BaseColor_1.png");
    if (aoMap == 0) {
        std::cerr << "Failed to load Ambient Occlusion (AO) texture!" << std::endl;
        return -1;
    }

    GLuint heightMap = loadTexture("../src/utils/Solar Panel_Stand_BaseColor_7.png");
    if (heightMap == 0) {
        std::cerr << "Failed to load Height Map texture!" << std::endl;
        return -1;
    }

    GLuint emissiveMap = loadTexture("../src/utils/Solar:metallic_texture-Solar:roughness_texture_5@channels=B.png");
    if (emissiveMap == 0) {
        std::cerr << "Failed to load Emissive Map texture!" << std::endl;
        return -1;
    }

    GLuint opacityMap = loadTexture("../src/utils/Solar:metallic_texture-Solar:roughness_texture_5@channels=G.png");
    if (opacityMap == 0) {
        std::cerr << "Failed to load Opacity Map texture!" << std::endl;
        return -1;
    }

    GLuint specularMap = loadTexture("../src/utils/Solar Panel_Stand_Metallic-Solar Panel_Stand_Roughness_8@cha.png");
    if (specularMap == 0) {
        std::cerr << "Failed to load Specular Map texture!" << std::endl;
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

    GLuint solarPanelShader = LoadShadersFromFile("../src/shader/solarPanel.vert", "../src/shader/solarPanel.frag");
    if (solarPanelShader == 0) {
        std::cerr << "Failed to load solar panel shaders." << std::endl;
        return -1;
    }

    GLuint haloShader = LoadShadersFromFile("../src/shader/halo.vert", "../src/shader/halo.frag");
     if (haloShader == 0) {
        std::cerr << "Failed to load halo shaders." << std::endl;
        return -1;
    }

    GLuint shadowShader = LoadShadersFromFile("../src/shader/shadow.vert", "../src/shader/shadow.frag");
     if (shadowShader == 0) {
        std::cerr << "Failed to load shadow shaders." << std::endl;
        return -1;
    }

    GLuint skyShader = LoadShadersFromFile("../src/shader/sky.vert", "../src/shader/sky.frag");
    if (skyShader == 0) {
        std::cerr << "Failed to load sky shaders." << std::endl;
        return -1;
    }

    GLuint sunVAO = createSunVAO();
    GLuint haloQuadVAO = createHaloQuadVAO();
    GLuint skyQuadVAO = createSkyQuadVAO();

    keepLoadingChunks = true;
    chunkThread = std::thread(chunkLoadingTask);

    updateChunks(currentChunkX, currentChunkZ);

    while (!chunkRequests.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        pollLoadedChunks();
    }

    glm::mat4 projectionMatrix = glm::perspective(glm::radians(FoV), 1024.0f / 768.0f, zNear, zFar);

    float orthoSize = 7000.0f; // Adjust as needed to fit your scene
    glm::mat4 lightProjection = glm::ortho( -orthoSize, orthoSize, -orthoSize, orthoSize, 0.1f, 5000.0f);
    glm::vec3 lightPos = eye_center - sunlightDirection * 1000.0f; 
    glm::mat4 lightView = glm::lookAt(lightPos, lightPos + sunlightDirection, glm::vec3(0, 1, 0));
    glm::mat4 lightSpaceMatrix = lightProjection * lightView;

    glEnable(GL_DEPTH_TEST);

    Turbine turbine = loadTurbine("../src/model/turbine/Turbine.glb");

    SolarPanel solarPanel = loadSolarPanel("../src/model/solarpanel/SolarPanel.glb");

    generateTurbineInstances();
    generateSolarPanelInstances(20);

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

    glGenBuffers(1, &solarPanelInstanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, solarPanelInstanceVBO);
    glBufferData(GL_ARRAY_BUFFER, solarPanelInstances.size() * sizeof(glm::mat4), &solarPanelInstances[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    for (auto& mesh : solarPanel.meshes) {
        glBindVertexArray(mesh.VAO);
        glBindBuffer(GL_ARRAY_BUFFER, solarPanelInstanceVBO);

        for (int i = 0; i < 4; i++) {
            glEnableVertexAttribArray(3 + i);
            glVertexAttribPointer(3 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(i * sizeof(glm::vec4)));
            glVertexAttribDivisor(3 + i, 1);
        }
        glBindVertexArray(0);
    }

    glClearColor(0.5f, 0.7f, 1.0f, 1.0f);

    double frameStartTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        // Frame start
        float currentFrameTime = glfwGetTime();
        deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;

        double currentTime = glfwGetTime();
        nbFrames++;
        if (currentTime - lastTime >= 1.0) { 
            double fps = double(nbFrames);
            std::string title = "Towards a Futuristic Emerald Isle. FPS: " + std::to_string(fps);
            glfwSetWindowTitle(window, title.c_str()); 
            nbFrames = 0;
            lastTime += 1.0;
        }

        processInput(window, deltaTime);
        pollLoadedChunks();

        glm::mat4 viewMatrix = glm::lookAt(eye_center, lookat, up);
        glm::mat4 vpMatrix = projectionMatrix * viewMatrix;

        glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glClear(GL_DEPTH_BUFFER_BIT);

        glUseProgram(shadowShader);
        GLuint lightSpaceLoc = glGetUniformLocation(shadowShader, "lightSpaceMatrix");
        glUniformMatrix4fv(lightSpaceLoc, 1, GL_FALSE, &lightSpaceMatrix[0][0]);

        {
            GLint modelLoc = glGetUniformLocation(shadowShader, "model");
            for (const auto& chunk : activeChunks) {
                int lodIndex = 0; 
                glm::mat4 terrainModel = glm::mat4(1.0f);
                glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &terrainModel[0][0]);
                const LODLevel& lodLevel = chunk.lodLevels[lodIndex];
                glBindVertexArray(lodLevel.VAO);
                glDrawElements(GL_TRIANGLES, lodLevel.indexCount, GL_UNSIGNED_INT, nullptr);
            }
        }

        {
            GLint modelLoc = glGetUniformLocation(shadowShader, "model");
            glm::mat4 identityModel = glm::mat4(1.0f);
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &identityModel[0][0]);

            for (size_t i = 0; i < turbine.meshes.size(); ++i) {
                glBindVertexArray(turbine.meshes[i].VAO);

                if (turbine.meshes[i].indexCount > 0) {
                    glDrawElementsInstanced(
                        GL_TRIANGLES,
                        turbine.meshes[i].indexCount,
                        turbine.meshes[i].indexType,
                        0,
                        NUM_TURBINES
                    );
                } else {
                    glDrawArraysInstanced(
                        GL_TRIANGLES,
                        0,
                        turbine.meshes[i].vertexCount,
                        NUM_TURBINES
                    );
                }
            }
        }

        {
            GLint modelLoc = glGetUniformLocation(shadowShader, "model");
            glm::mat4 identityModel = glm::mat4(1.0f);
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &identityModel[0][0]);

            for (const auto& mesh : solarPanel.meshes) {
                glBindVertexArray(mesh.VAO);

                if (mesh.indexCount > 0) {
                    glDrawElementsInstanced(
                        GL_TRIANGLES,
                        mesh.indexCount,
                        mesh.indexType,
                        0,
                        static_cast<GLsizei>(solarPanelInstances.size())
                    );
                } else {
                    glDrawArraysInstanced(
                        GL_TRIANGLES,
                        0,
                        mesh.vertexCount,
                        static_cast<GLsizei>(solarPanelInstances.size())
                    );
                }
            }
        }

        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        int windowWidth, windowHeight;
        glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
        glViewport(0, 0, windowWidth, windowHeight);
        
        glClear(GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glUseProgram(skyShader);
        glBindVertexArray(skyQuadVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glEnable(GL_DEPTH_TEST);

        renderTerrainChunks(terrainShader, vpMatrix, grassTexture, lightSpaceMatrix, depthMap);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        renderSun(sunLightingShader, sunVAO, vpMatrix);
        renderHalo(haloShader, haloQuadVAO, vpMatrix);
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
        renderTurbine(turbine, turbineShader, vpMatrix, lightSpaceMatrix, depthMap);
        renderSolarPanels(solarPanel, solarPanelShader, vpMatrix, baseColor, normalMap, metallicMap, roughnessMap, aoMap, heightMap, emissiveMap, opacityMap, specularMap, lightSpaceMatrix, depthMap);

        glfwSwapBuffers(window);
        glfwPollEvents();

        double frameEndTime = glfwGetTime();
        double frameDuration = frameEndTime - frameStartTime;
        frameStartTime = glfwGetTime(); 
    }

    glfwTerminate();

    // Signal thread to stop, then join
    keepLoadingChunks = false;
    if (chunkThread.joinable()) {
        chunkThread.join();
    }

    return 0;
}

void renderTerrainChunks(GLuint shader, const glm::mat4& vpMatrix, GLuint texture, glm::mat4 lightSpaceMatrix, GLuint depthMap)
{
    glUseProgram(shader);

    GLint vpLoc = glGetUniformLocation(shader, "vpMatrix");
    glUniformMatrix4fv(vpLoc, 1, GL_FALSE, &vpMatrix[0][0]);

    GLint lightSpaceLoc = glGetUniformLocation(shader, "lightSpaceMatrix");
    glUniformMatrix4fv(lightSpaceLoc, 1, GL_FALSE, &lightSpaceMatrix[0][0]);

    glActiveTexture(GL_TEXTURE9);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    GLint shadowMapLoc = glGetUniformLocation(shader, "shadowMap");
    glUniform1i(shadowMapLoc, 9);

    glUniform3fv(glGetUniformLocation(shader, "lightDir"), 1, &sunlightDirection[0]);
    glUniform3fv(glGetUniformLocation(shader, "lightColor"), 1, &sunlightColor[0]);
    glUniform3f(glGetUniformLocation(shader, "viewPos"),
                eye_center.x, eye_center.y, eye_center.z);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glUniform1i(glGetUniformLocation(shader, "terrainTexture"), 0);

    GLint modelMatrixLoc = glGetUniformLocation(shader, "modelMatrix");

    for (const auto& chunk : activeChunks) {
        glm::vec3 chunkCenter(
            chunk.position.x + (GRID_SIZE * GRID_SCALE * 0.5f),
            0.0f,
            chunk.position.y + (GRID_SIZE * GRID_SCALE * 0.5f)
        );
        float distance = glm::distance(chunkCenter, eye_center);
        int lodIndex = getLODIndex(distance);
        if (lodIndex < 0) lodIndex = 0;
        if (lodIndex >= (int)chunk.lodLevels.size()) {
            lodIndex = (int)chunk.lodLevels.size() - 1;
        }
        const LODLevel& lodLevel = chunk.lodLevels[lodIndex];

        glm::mat4 chunkModel = glm::translate(
            glm::mat4(1.0f),
            glm::vec3(chunk.position.x, 0.0f, chunk.position.y)
        );

        glUniformMatrix4fv(modelMatrixLoc, 1, GL_FALSE, &chunkModel[0][0]);

        glBindVertexArray(lodLevel.VAO);
        glDrawElements(GL_TRIANGLES, lodLevel.indexCount, GL_UNSIGNED_INT, 0);
    }
}


void renderSun(GLuint shader, GLuint sunVAO, const glm::mat4& vpMatrix) {
    glUseProgram(shader);

    float forwardDistance = 200.0f; 
    float rightOffset     = 75.0f; 
    float upOffset        = 50.0f; 
    glm::vec3 sunPosition = eye_center + forwardDirection * forwardDistance + rightDirection * rightOffset + up * upOffset;

    glm::mat4 model = glm::translate(glm::mat4(1.0f), sunPosition);
    model = glm::scale(model, glm::vec3(7.5f));

    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &model[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shader, "vpMatrix"), 1, GL_FALSE, &vpMatrix[0][0]);

    glm::vec3 brightSunColor = glm::vec3(1.0f, 0.98f, 0.90f);
    glUniform3fv(glGetUniformLocation(shader, "lightColor"), 1, &brightSunColor[0]);
    glUniform1f(glGetUniformLocation(shader, "intensity"), 5.0f);

    glm::vec3 dir = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));
    glUniform3fv(glGetUniformLocation(shader, "lightDir"), 1, &dir[0]);

    glBindVertexArray(sunVAO);
    glDrawElements(GL_TRIANGLES, 36 * 18 * 6, GL_UNSIGNED_INT, 0);
}

void renderHalo(GLuint shader, GLuint haloQuadVAO, const glm::mat4& vpMatrix) {
    glUseProgram(shader);

    float forwardDistance = 200.0f;
    float rightOffset = 75.0f;
    float upOffset = 50.0f;
    glm::vec3 sunPos = eye_center 
                       + forwardDirection * forwardDistance
                       + rightDirection   * rightOffset
                       + up               * upOffset;

    glm::mat4 billboard = glm::mat4(1.0f);
    billboard[0] = glm::vec4(rightDirection, 0.0f);  
    billboard[1] = glm::vec4(up, 0.0f);              
    billboard[2] = glm::vec4(-forwardDirection, 0.0f); 

    glm::mat4 modelHalo = glm::translate(glm::mat4(1.0f), sunPos) 
                        * billboard
                        * glm::scale(glm::mat4(1.0f), glm::vec3(50.0f)); 

    // Set uniforms
    glUniformMatrix4fv(glGetUniformLocation(shader, "vpMatrix"), 1, GL_FALSE, &vpMatrix[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &modelHalo[0][0]);

    glm::vec3 haloColor = glm::vec3(1.0f, 0.95f, 0.8f); 
    glUniform3fv(glGetUniformLocation(shader, "haloColor"), 1, &haloColor[0]);

    glUniform1f(glGetUniformLocation(shader, "haloAlpha"), 0.3f);  
    glUniform1f(glGetUniformLocation(shader, "haloIntensity"), 1.0f);  

    // Draw the halo quad
    glBindVertexArray(haloQuadVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void renderTurbine(const Turbine& turbine, GLuint shader, const glm::mat4& vpMatrix, glm::mat4 lightSpaceMatrix, GLuint depthMap) {
    glUseProgram(shader);

    GLint lightSpaceLoc = glGetUniformLocation(shader, "lightSpaceMatrix");
    glUniformMatrix4fv(lightSpaceLoc, 1, GL_FALSE, &lightSpaceMatrix[0][0]);
    glActiveTexture(GL_TEXTURE9);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    GLint shadowMapLoc = glGetUniformLocation(shader, "shadowMap");
    glUniform1i(shadowMapLoc, 9);

    static float bladeRotation = 0.0f;
    float rotationSpeed = 0.10f;
    bladeRotation += glfwGetTime() * rotationSpeed;
    bladeRotation = fmod(bladeRotation, 360.0f);

    glm::mat4 baseModelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(50.0f, -5.0f, 50.0f));
    baseModelMatrix = glm::rotate(baseModelMatrix, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    baseModelMatrix = glm::scale(baseModelMatrix, glm::vec3(1.0f));

    glm::vec3 bladeAttachmentPoint(0.0f, 70.0f, 0.0f);
    glm::vec3 rotationCircleScale(0.5f, 0.5f, 0.5f);

    for (size_t i = 0; i < turbine.meshes.size(); ++i) {
        glm::mat4 modelMatrix = baseModelMatrix;

        if (i == 16) { // Blade mesh
            modelMatrix = glm::translate(modelMatrix, bladeAttachmentPoint);
            modelMatrix = glm::scale(modelMatrix, rotationCircleScale);
            modelMatrix = glm::rotate(modelMatrix, glm::radians(bladeRotation), glm::vec3(0.0f, 0.0f, 1.0f));
            modelMatrix = glm::scale(modelMatrix, glm::vec3(1.0f) / rotationCircleScale);
            modelMatrix = glm::translate(modelMatrix, -bladeAttachmentPoint);
        }

        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &modelMatrix[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shader, "vpMatrix"), 1, GL_FALSE, &vpMatrix[0][0]);
        glUniform3f(glGetUniformLocation(shader, "lightColor"), 1.0f, 1.0f, 1.0f);
        glUniform3f(glGetUniformLocation(shader, "lightDir"), -1.0f, -1.0f, -1.0f);
        glUniform3f(glGetUniformLocation(shader, "viewPos"), eye_center.x, eye_center.y, eye_center.z);
        glUniform1i(glGetUniformLocation(shader, "isBlade"), (i == 16) ? 1 : 0);

        glBindVertexArray(turbine.meshes[i].VAO);

        if (turbine.meshes[i].indexCount > 0) {
            glDrawElementsInstanced(GL_TRIANGLES, turbine.meshes[i].indexCount, turbine.meshes[i].indexType, 0, NUM_TURBINES);
        } else {
            glDrawArraysInstanced(GL_TRIANGLES, 0, turbine.meshes[i].vertexCount, NUM_TURBINES);
        }
    }
}

void renderSolarPanels(const SolarPanel& solarPanel, GLuint shader, const glm::mat4& vpMatrix,
                       GLuint baseColor, GLuint normalMap, GLuint metallicMap, GLuint roughnessMap,
                       GLuint aoMap, GLuint heightMap, GLuint emissiveMap, GLuint opacityMap, GLuint specularMap, glm::mat4 lightSpaceMatrix, GLuint depthMap) {
    glUseProgram(shader);
    GLint lightSpaceLoc = glGetUniformLocation(shader, "lightSpaceMatrix");
    glUniformMatrix4fv(lightSpaceLoc, 1, GL_FALSE, &lightSpaceMatrix[0][0]);

    glActiveTexture(GL_TEXTURE9);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    GLint shadowMapLoc = glGetUniformLocation(shader, "shadowMap");
    glUniform1i(shadowMapLoc, 9);
    glUniform1f(glGetUniformLocation(shader, "normalBlendFactor"), 1.0f);
    glUniform3fv(glGetUniformLocation(shader, "viewPos"), 1, &eye_center[0]);
    glUniform3fv(glGetUniformLocation(shader, "lightDir"), 1, &sunlightDirection[0]);
    glUniform3fv(glGetUniformLocation(shader, "lightColor"), 1, &sunlightColor[0]);

    // Pass the VP matrix
    GLint vpMatrixLoc = glGetUniformLocation(shader, "vpMatrix");
    glUniformMatrix4fv(vpMatrixLoc, 1, GL_FALSE, &vpMatrix[0][0]);

    // Bind textures and set uniforms
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, baseColor);
    glUniform1i(glGetUniformLocation(shader, "baseColorMap"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normalMap);
    glUniform1i(glGetUniformLocation(shader, "normalMap"), 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, metallicMap);
    glUniform1i(glGetUniformLocation(shader, "metallicMap"), 2);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, roughnessMap);
    glUniform1i(glGetUniformLocation(shader, "roughnessMap"), 3);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, aoMap);
    glUniform1i(glGetUniformLocation(shader, "aoMap"), 4);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, heightMap);
    glUniform1i(glGetUniformLocation(shader, "heightMap"), 5);

    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, emissiveMap);
    glUniform1i(glGetUniformLocation(shader, "emissiveMap"), 6);

    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_2D, opacityMap);
    glUniform1i(glGetUniformLocation(shader, "opacityMap"), 7);

    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, specularMap);
    glUniform1i(glGetUniformLocation(shader, "specularMap"), 8);

    // Render each mesh
    for (const auto& mesh : solarPanel.meshes) {
        glBindVertexArray(mesh.VAO);
        if (mesh.indexCount > 0) {
            glDrawElementsInstanced(GL_TRIANGLES, mesh.indexCount, mesh.indexType, 0, static_cast<GLsizei>(solarPanelInstances.size()));
        } else {
            glDrawArraysInstanced(GL_TRIANGLES, 0, mesh.vertexCount, static_cast<GLsizei>(solarPanelInstances.size()));
        }
    }
}

int getLODIndex(float distance)
{
    if (distance < 400.0f)
        return 0;  // Highest detail
    else if (distance < 800.0f)
        return 1;  // Medium detail
    else
        return 2;  // Lowest detail
}

void generateTurbineInstances()
{
    turbineInstances.clear();
    turbineInstances.reserve(NUM_TURBINES);

    srand(42);

    float rangeX = 2000.0f;
    float rangeZ = 2000.0f;

    for (int i = 0; i < NUM_TURBINES; i++)
    {
        float x = static_cast<float>(rand()) / RAND_MAX * rangeX;
        float z = static_cast<float>(rand()) / RAND_MAX * rangeZ;

        float y = getTerrainHeight(x, z);

        glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));

        float angle = glm::radians(static_cast<float>(rand() % 360));
        model = glm::rotate(model, angle, glm::vec3(0,1,0));

        turbineInstances.push_back(model);
    }
}

void generateSolarPanelInstances(int panelCount)
{
    solarPanelInstances.clear();
    solarPanelInstances.reserve(panelCount);

    srand(123);

    float rangeX = 2000.0f;
    float rangeZ = 2000.0f;
    float verticalOffset = 25.0f;

    for (int i = 0; i < panelCount; i++)
    {
        float x = static_cast<float>(rand()) / RAND_MAX * rangeX;
        float z = static_cast<float>(rand()) / RAND_MAX * rangeZ;

        float y = getTerrainHeight(x, z) + verticalOffset;
        glm::vec3 panelPosition(x, y, z);

        glm::vec3 toCamera = glm::normalize(eye_center - panelPosition);
        float angleY = atan2(toCamera.x, toCamera.z);

        glm::mat4 model(1.0f);
        model = glm::translate(model, panelPosition);
        model = glm::rotate(model, angleY, glm::vec3(0, 1, 0));

        model = glm::rotate(model, glm::radians(-30.0f), glm::vec3(1, 0, 0));
        model = glm::scale(model, glm::vec3(0.5f));

        solarPanelInstances.push_back(model);
    }

    if (solarPanelInstanceVBO != 0) {
        glBindBuffer(GL_ARRAY_BUFFER, solarPanelInstanceVBO);
        glBufferData(GL_ARRAY_BUFFER,
                     solarPanelInstances.size() * sizeof(glm::mat4),
                     &solarPanelInstances[0],
                     GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void updateChunks(int cx, int cz)
{
    // Our chosen "load distance" for chunks:
    int range = 10;
    int startX = cx - range;
    int endX   = cx + range;
    int startZ = cz - range;
    int endZ   = cz + range;

    // 1) Remove out-of-range chunks
    activeChunks.erase(
        std::remove_if(activeChunks.begin(), activeChunks.end(),
            [=](const Chunk &chunk)
            {
                return (chunk.chunkX < startX || chunk.chunkX > endX ||
                        chunk.chunkZ < startZ || chunk.chunkZ > endZ);
            }
        ),
        activeChunks.end()
    );

    // 2) Enqueue requests for chunks in range that we do not already have
    for (int z = startZ; z <= endZ; ++z) {
        for (int x = startX; x <= endX; ++x) {
            bool found = false;
            for (const auto &chunk : activeChunks) {
                if (chunk.chunkX == x && chunk.chunkZ == z) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::lock_guard<std::mutex> lock(chunkMutex);
                chunkRequests.push({x, z});
            }
        }
    }
}




std::vector<Vertex> generateTerrain(unsigned int gridSize, float gridScale, float heightScale, 
                                    std::vector<unsigned int>& indices, int chunkX, int chunkZ)
{
    std::vector<Vertex> vertices;
    vertices.reserve((gridSize + 1) * (gridSize + 1));
    
    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    noise.SetFractalType(FastNoiseLite::FractalType_FBm);
    noise.SetFractalOctaves(6);
    noise.SetFrequency(0.02f);
    noise.SetFractalLacunarity(2.0f);
    noise.SetFractalGain(0.5f);

    float worldOffsetX = chunkX * (float)gridSize * gridScale;
    float worldOffsetZ = chunkZ * (float)gridSize * gridScale;

    for (unsigned int z = 0; z <= gridSize; ++z) {
        for (unsigned int x = 0; x <= gridSize; ++x) {
            float localX = x * gridScale;
            float localZ = z * gridScale;

            float globalX = worldOffsetX + localX;
            float globalZ = worldOffsetZ + localZ;

            float lowFrequencyNoise = noise.GetNoise(globalX * 0.05f, globalZ * 0.05f);
            float midFrequencyNoise = noise.GetNoise(globalX * 0.2f, globalZ * 0.2f);
            float highFrequencyNoise = noise.GetNoise(globalX * 0.8f, globalZ * 0.8f);

            float biomeFactor = (noise.GetNoise(globalX * 0.01f, globalZ * 0.01f) + 1.0f) / 2.0f;
            float biomeHeightScale = glm::mix(20.0f, 60.0f, biomeFactor);

            float height = ((lowFrequencyNoise * 0.5f + 
                             midFrequencyNoise * 0.3f + 
                             highFrequencyNoise * 0.2f) + 1.0f) * 0.5f * biomeHeightScale;

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

float getTerrainHeight(float globalX, float globalZ)
{
    static FastNoiseLite noise;
    static bool noiseInitialized = false;
    if (!noiseInitialized) {
        noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        noise.SetFractalType(FastNoiseLite::FractalType_FBm);
        noise.SetFractalOctaves(6);
        noise.SetFrequency(0.02f);
        noise.SetFractalLacunarity(2.0f);
        noise.SetFractalGain(0.5f);
        noiseInitialized = true;
    }

    float lowFrequencyNoise = noise.GetNoise(globalX * 0.05f, globalZ * 0.05f);
    float midFrequencyNoise = noise.GetNoise(globalX * 0.2f,  globalZ * 0.2f);
    float highFrequencyNoise= noise.GetNoise(globalX * 0.8f,  globalZ * 0.8f);

    float biomeFactor = (noise.GetNoise(globalX * 0.01f, globalZ * 0.01f) + 1.0f)*0.5f;
    float biomeHeightScale = glm::mix(20.0f, 60.0f, biomeFactor);

    float height = ((lowFrequencyNoise * 0.5f +
                     midFrequencyNoise * 0.3f +
                     highFrequencyNoise * 0.2f) + 1.0f) * 0.5f 
                     * biomeHeightScale;

    return height;
}

void chunkLoadingTask()
{
    std::vector<unsigned int> lodGridSizes = { 100, 50, 25 };

    while (keepLoadingChunks)
    {
        std::pair<int,int> request;
        {
            std::lock_guard<std::mutex> lock(chunkMutex);
            if (!chunkRequests.empty())
            {
                request = chunkRequests.front();
                chunkRequests.pop();
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        }

        int x = request.first;
        int z = request.second;

        glm::vec2 chunkPos(
            x * GRID_SIZE * GRID_SCALE,
            z * GRID_SIZE * GRID_SCALE
        );

        std::vector<ChunkData> allLODData;
        allLODData.reserve(lodGridSizes.size());

        for (auto lodGrid : lodGridSizes)
        {
            std::vector<unsigned int> indices;
            std::vector<Vertex> vertices = generateTerrain(
                lodGrid, // <--- smaller or bigger
                GRID_SCALE * (static_cast<float>(GRID_SIZE) / lodGrid),
                HEIGHT_SCALE,
                indices,
                x, z
            );

            ChunkData cd;
            cd.vertices  = std::move(vertices);
            cd.indices   = std::move(indices);
            cd.position  = chunkPos;
            cd.chunkX    = x;
            cd.chunkZ    = z;

            allLODData.push_back(cd);
        }

        {
            std::lock_guard<std::mutex> lock(chunkMutex);
            chunkDataQueue.push(allLODData);
        }
    }
}

void processInput(GLFWwindow *window, float deltaTime) {
    static float baseSpeed = 25.0f;
    float movementSpeed = baseSpeed * deltaTime;
    glm::vec3 movement(0.0f);

    glm::vec3 flatForward = glm::normalize(glm::vec3(forwardDirection.x, 0.0f, forwardDirection.z));
    glm::vec3 flatRight = glm::normalize(glm::vec3(rightDirection.x, 0.0f, rightDirection.z));

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        movement += flatForward * movementSpeed;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        movement -= flatForward * movementSpeed;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        movement -= flatRight * movementSpeed;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        movement += flatRight * movementSpeed;

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
