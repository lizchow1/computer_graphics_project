#version 330 core

layout (location = 0) in vec2 aPos;

out vec2 vUV;

void main()
{
    vUV = (aPos + 1.0) * 0.5; 
    gl_Position = vec4(aPos, 0.0, 1.0);
}
