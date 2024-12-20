#version 330 core

in vec3 fragPosition;
in vec3 fragNormal;

uniform vec3 lightColor;
uniform vec3 lightDir;
uniform vec3 viewPos;

out vec4 fragColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 light = normalize(-lightDir);
    float diff = max(dot(normal, light), 0.0);

    vec3 diffuse = diff * lightColor;

    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 reflectDir = reflect(-light, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;

    vec3 result = diffuse + specular;
    fragColor = vec4(result, 1.0);
}
