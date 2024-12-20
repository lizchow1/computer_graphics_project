#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 vpMatrix;

out vec3 fragPosition;
out vec3 fragNormal;

void main() {
    fragPosition = vec3(model * vec4(aPosition, 1.0));
    fragNormal = mat3(transpose(inverse(model))) * aNormal;

    gl_Position = vpMatrix * vec4(fragPosition, 1.0);
}