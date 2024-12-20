#version 330 core

layout(location = 0) in vec3 aPos;            
layout(location = 1) in vec3 aNormal;         
layout(location = 2) in vec2 aTexCoords;      
layout(location = 3) in mat4 instanceMatrix;  

out vec3 fragPosition;   
out vec3 fragNormal;     

uniform mat4 model;
uniform mat4 vpMatrix;

void main() {
    mat4 worldMatrix = instanceMatrix * model;
    vec4 worldPos = worldMatrix * vec4(aPos, 1.0);

    fragPosition = worldPos.xyz;
    fragNormal = mat3(transpose(inverse(worldMatrix))) * aNormal;

    gl_Position = vpMatrix * worldPos;
}
