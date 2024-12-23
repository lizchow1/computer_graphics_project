#version 330 core

in vec3 FragPos;
in vec3 Normal;       // Geometry normal
in vec2 TexCoords;    // Primary UV

out vec4 FragColor;

uniform sampler2D baseColorMap; // Base color texture
uniform sampler2D normalMap;    // Normal map texture

// Lighting uniforms
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 viewPos;

// Uniform for blending factor (0.0 = pure geometry normal, 1.0 = pure normal map)
uniform float normalBlendFactor;

void main() {
    // Sample base color texture
    vec3 albedo = texture(baseColorMap, TexCoords).rgb;

    // Sample normal map and transform to range [-1, 1]
    vec3 normalMapNormal = texture(normalMap, TexCoords).rgb;
    normalMapNormal = normalize(normalMapNormal * 2.0 - 1.0);

    // Blend geometry normal with the normal map normal
    vec3 blendedNormal = normalize(mix(Normal, normalMapNormal, normalBlendFactor));

    // Lighting calculations
    vec3 norm = normalize(blendedNormal);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-normalize(lightDir), norm);

    // Diffuse lighting
    float diff = max(dot(norm, normalize(lightDir)), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular lighting (Phong model)
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0); // Shininess = 32
    vec3 specular = spec * lightColor;

    // Combine lighting with albedo
    vec3 lighting = albedo * (diffuse + specular);

    FragColor = vec4(lighting, 1.0);
}
