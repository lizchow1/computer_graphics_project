#version 330 core

layout(location = 0) in vec3 inPosition;
layout(location = 3) in mat4 instanceMatrix;

uniform mat4 lightSpaceMatrix; 
uniform mat4 model;

void main()
{
    gl_Position = lightSpaceMatrix * model * instanceMatrix * vec4(inPosition, 1.0);
}