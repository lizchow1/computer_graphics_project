#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 vpMatrix;

out vec3 fragNormal;

void main() {
    vec4 worldPos = model * vec4(aPos, 1.0);
    gl_Position = vpMatrix * worldPos;

    fragNormal = mat3(transpose(inverse(model))) * aNormal; 
}
