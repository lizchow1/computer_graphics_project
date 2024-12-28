#version 330 core

in vec2 fragTexCoords;
in vec3 fragNormal;
in vec4 fragPosLightSpace;

out vec4 fragColor;

uniform sampler2D terrainTexture; 
uniform sampler2D shadowMap;

uniform mat4 lightSpaceMatrix;
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 viewPos;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z < 0.0 || projCoords.z > 1.0)
    {
        return 0.0;
    }

    float currentDepth = projCoords.z;
    float closestDepth = texture(shadowMap, projCoords.xy).r;

    float bias = 0.005;  

    float shadow = (currentDepth - bias > closestDepth) ? 1.0 : 0.0;
    return shadow;
}

void main()
{
    vec3 albedo = texture(terrainTexture, fragTexCoords).rgb;

    vec3 normal = normalize(fragNormal);
    vec3 L = normalize(-lightDir);     
    float diff = max(dot(normal, L), 0.0);
    
    float shadow = ShadowCalculation(fragPosLightSpace);

    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * albedo;

    vec3 lighting = (1.0 - shadow) * (diff * albedo * lightColor);

    vec3 color = ambient + lighting;

    fragColor = vec4(color, 1.0);
}
