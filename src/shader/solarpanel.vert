#version 330 core

// Vertex attributes from VAO
layout(location = 0) in vec3 aPos;       // Position
layout(location = 1) in vec3 aNormal;    // Normal
layout(location = 2) in vec2 aTexCoords; // Texture coordinates

// Instance matrix (divided across 4 locations)
layout(location = 3) in mat4 instanceMatrix;

// Outputs to the fragment shader
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;

// Uniforms
uniform mat4 vpMatrix; // Combined view-projection matrix

void main() {
    // Transform position and normal
    FragPos = vec3(instanceMatrix * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(instanceMatrix))) * aNormal;
    TexCoords = aTexCoords;

    // Final position in clip space
    gl_Position = vpMatrix * vec4(FragPos, 1.0);
}
