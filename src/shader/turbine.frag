#version 330 core

in vec3 fragPosition;
in vec3 fragNormal;

uniform vec3 lightColor;
uniform vec3 lightDir;
uniform vec3 viewPos;

uniform sampler2D shadowMap;
uniform mat4 lightSpaceMatrix;

out vec4 fragColor;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    if(projCoords.x < 0.0 || projCoords.x > 1.0 ||
       projCoords.y < 0.0 || projCoords.y > 1.0 ||
       projCoords.z < 0.0 || projCoords.z > 1.0)
    {
       return 0.0;
    }

    float currentDepth = projCoords.z;
    float closestDepth = texture(shadowMap, projCoords.xy).r;


    float bias = 0.005;
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    return shadow;
}

void main() 
{
    vec3 normal = normalize(fragNormal);
    vec3 light = normalize(-lightDir);
    float diff = max(dot(normal, light), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 reflectDir = reflect(-light, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;

    vec4 fragPosLightSpace = lightSpaceMatrix * vec4(fragPosition, 1.0);
    float shadow = ShadowCalculation(fragPosLightSpace);

    vec3 color = (1.0 - shadow) * (diffuse + specular);
    color += 0.2 * lightColor * (1.0 - shadow);

    fragColor = vec4(color, 1.0);
}