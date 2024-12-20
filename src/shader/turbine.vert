#version 330 core
layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;

layout (location = 3) in vec4 inInstMatRow0;
layout (location = 4) in vec4 inInstMatRow1;
layout (location = 5) in vec4 inInstMatRow2;
layout (location = 6) in vec4 inInstMatRow3;

uniform mat4 vpMatrix;
uniform mat4 bladeRotationMatrix;
uniform int isBlade;

out vec3 fragPosition;
out vec3 fragNormal;

void main() {
    mat4 instanceModel = mat4(inInstMatRow0, inInstMatRow1, inInstMatRow2, inInstMatRow3);
    mat4 finalModel = instanceModel;

    if (isBlade == 1) {
        finalModel = finalModel * bladeRotationMatrix;
    }

    vec4 worldPos = finalModel * vec4(inPosition, 1.0);
    gl_Position = vpMatrix * worldPos;

    fragPosition = worldPos.xyz;
    mat3 normalMatrix = mat3(transpose(inverse(finalModel)));
    fragNormal = normalize(normalMatrix * inNormal);
}