#version 330 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoords;

uniform mat4 vpMatrix;         
uniform mat4 modelMatrix;      
uniform mat4 lightSpaceMatrix; 

out vec2 fragTexCoords;
out vec3 fragNormal;
out vec4 fragPosLightSpace;

void main()
{
    vec4 worldPos = modelMatrix * vec4(inPosition, 1.0);

    fragNormal = mat3(transpose(inverse(modelMatrix))) * inNormal;

    fragPosLightSpace = lightSpaceMatrix * worldPos;

    fragTexCoords = inTexCoords;

    gl_Position = vpMatrix * worldPos;
}