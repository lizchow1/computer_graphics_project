#version 330 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoords;

uniform mat4 vpMatrix;

out vec2 fragTexCoords;

void main() {
    gl_Position = vpMatrix * vec4(inPosition, 1.0);
    fragTexCoords = inTexCoords; 
}