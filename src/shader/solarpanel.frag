#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

out vec4 FragColor;

// All your texture samplers:
uniform sampler2D baseColorMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform sampler2D heightMap;
uniform sampler2D emissiveMap;
uniform sampler2D opacityMap;
uniform sampler2D specularMap;

// Lighting
uniform vec3 lightDir;   // e.g. sunlightDirection
uniform vec3 lightColor; // e.g. sunlightColor
uniform vec3 viewPos;    // camera eye position

// Simple normal-blend factor
uniform float normalBlendFactor; // e.g. set this to 1.0 to rely on normalMap

void main()
{
    // Base color
    vec3 albedo = texture(baseColorMap, TexCoords).rgb;

    // Normal map in [0..1], transform to [-1..1]
    vec3 normalMapSample = texture(normalMap, TexCoords).rgb; 
    vec3 normalTangentSpace = normalMapSample * 2.0 - 1.0;

    // Without tangents, we just do a naive blend with the geometry normal
    vec3 N = normalize( mix(Normal, normalTangentSpace, normalBlendFactor) );

    // Metallic/roughness/ao maps
    float metallic  = texture(metallicMap, TexCoords).r;
    float roughness = texture(roughnessMap, TexCoords).r;
    float ao        = texture(aoMap, TexCoords).r;

    // Optionally sample these if you do real PBR or if you want an effect:
    // float height   = texture(heightMap, TexCoords).r;
    // vec3 emissive = texture(emissiveMap, TexCoords).rgb;
    // float opacity  = texture(opacityMap, TexCoords).r;
    // float specular = texture(specularMap, TexCoords).r;

    // For a basic Blinn-Phong approach:
    vec3 L = normalize(-lightDir);
    vec3 V = normalize(viewPos - FragPos);
    vec3 R = reflect(-L, N);

    // Diffuse
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * diff * lightColor;

    // Specular (Blinn-Phong)
    vec3 H = normalize(L + V);
    float specAngle = max(dot(N, H), 0.0);
    float specPower = mix(64.0, 4.0, roughness); // roughness modifies exponent
    float specularFactor = pow(specAngle, specPower);
    vec3 specularColor = mix(vec3(0.04), albedo, metallic);
    vec3 specularTerm  = specularFactor * specularColor * lightColor;

    // Ambient occlusion
    vec3 ambient = albedo * 0.05 * ao;

    // Final color
    vec3 color = ambient + diffuse + specularTerm;

    FragColor = vec4(color, 1.0);
}
