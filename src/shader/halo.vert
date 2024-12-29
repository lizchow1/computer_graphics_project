#version 330 core

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec2 inUV;

out vec2 vUV;

uniform mat4 vpMatrix;
uniform mat4 model;

void main()
{
    vUV = inUV;
    gl_Position = vpMatrix * model * vec4(inPosition, 1.0);
}
