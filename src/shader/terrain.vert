#version 330 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoords;

uniform mat4 vpMatrix;
uniform mat4 modelMatrix;
uniform mat4 lightSpaceMatrix;
uniform vec2 chunkOffset;

out vec2 fragTexCoords;
out vec3 fragNormal;
out vec4 fragPosLightSpace;

void main() {
    vec3 worldPosition = vec3(inPosition.x + chunkOffset.x, inPosition.y, inPosition.z + chunkOffset.y);
    fragNormal = mat3(transpose(inverse(modelMatrix))) * inNormal;
    fragPosLightSpace = lightSpaceMatrix * vec4(vec3(modelMatrix * vec4(worldPosition, 1.0)), 1.0);
    fragTexCoords = inTexCoords;
    gl_Position = vpMatrix * vec4(worldPosition, 1.0);
}