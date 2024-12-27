#version 330 core

layout(location = 0) in vec3 aPos;       
layout(location = 1) in vec3 aNormal;    
layout(location = 2) in vec2 aTexCoords;

// Instance matrix
layout(location = 3) in mat4 instanceMatrix;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;

uniform mat4 vpMatrix;

void main()
{
    vec4 worldPos = instanceMatrix * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;

    // Just using object-space normal or minimal correction
    // For a better normal-map, you'd pass tangents/bitangents too.
    mat3 normalMatrix = mat3(transpose(inverse(instanceMatrix)));
    Normal = normalize(normalMatrix * aNormal);

    TexCoords = aTexCoords;
    gl_Position = vpMatrix * worldPos;
}
