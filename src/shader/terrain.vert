#version 330 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoords;

uniform mat4 vpMatrix;
uniform vec2 chunkOffset;

out vec2 fragTexCoords;

void main() {
    vec3 worldPosition = vec3(inPosition.x + chunkOffset.x, inPosition.y, inPosition.z + chunkOffset.y);
    gl_Position = vpMatrix * vec4(worldPosition, 1.0);
    fragTexCoords = inTexCoords;
}