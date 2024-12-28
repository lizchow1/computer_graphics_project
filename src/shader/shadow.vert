#version 330 core
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 3) in mat4 instanceMatrix;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main()
{
    vec4 worldPos = model * instanceMatrix * vec4(inPosition, 1.0);
    gl_Position = lightSpaceMatrix * worldPos;
}
