#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 vpMatrix;

out vec3 fragNormal;

void main() {
    vec4 worldPos = model * vec4(aPos, 1.0);
    gl_Position = vpMatrix * worldPos;

    // We need normals in world space; since model is a translation+scale, 
    // if scale is uniform, we can just transform normal by the inverse transpose of model.
    // For simplicity, if scale=uniform and no rotation needed, normal = aNormal is enough.
    fragNormal = mat3(transpose(inverse(model))) * aNormal; 
}
