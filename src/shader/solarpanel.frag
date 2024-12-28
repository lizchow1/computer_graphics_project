#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

out vec4 FragColor;

uniform sampler2D baseColorMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform sampler2D heightMap;
uniform sampler2D emissiveMap;
uniform sampler2D opacityMap;
uniform sampler2D specularMap;

uniform vec3 lightDir;   
uniform vec3 lightColor; 
uniform vec3 viewPos;    

uniform sampler2D shadowMap;
uniform mat4 lightSpaceMatrix;

uniform float normalBlendFactor; 

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
    float shadow = (currentDepth - bias > closestDepth) ? 1.0 : 0.0;
    return shadow;
}

void main()
{
    vec3 nMap = texture(normalMap, TexCoords).rgb * 2.0 - 1.0;
    vec3 N = normalize(mix(Normal, nMap, normalBlendFactor));

    // Basic PBR or Blinn-Phong (your existing logic)
    vec3 albedo = texture(baseColorMap, TexCoords).rgb;
    float metallic  = texture(metallicMap, TexCoords).r;
    float roughness = texture(roughnessMap, TexCoords).r;
    float ao        = texture(aoMap, TexCoords).r;
    // etc...

    vec3 L = normalize(-lightDir);
    vec3 V = normalize(viewPos - FragPos);
    float diff = max(dot(N, L), 0.0);

    // Simple Blinn-Phong for example
    vec3 diffuse  = diff * albedo * lightColor;
    vec3 H = normalize(L + V);
    float specAngle = max(dot(N, H), 0.0);
    float specPower = mix(64.0, 4.0, roughness);
    float specularFactor = pow(specAngle, specPower);
    vec3 specularColor   = mix(vec3(0.04), albedo, metallic);
    vec3 specularTerm    = specularFactor * specularColor * lightColor;

    // Shadow factor
    vec4 fragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
    float shadow = ShadowCalculation(fragPosLightSpace);

    // Combine lighting
    vec3 ambient = albedo * 0.05 * ao; 
    vec3 color   = ambient + (1.0 - shadow)*(diffuse + specularTerm);

    FragColor = vec4(color, 1.0);
}