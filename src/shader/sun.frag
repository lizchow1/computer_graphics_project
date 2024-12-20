#version 330 core
in vec3 fragNormal;
in vec3 fragPosition;
out vec4 FragColor;

uniform vec3 lightColor;
uniform float intensity;

uniform vec3 lightDir; 

void main() {
    vec3 N = normalize(fragNormal);

    float diff = max(dot(N, -lightDir), 0.0);

    vec3 color = lightColor * (intensity * 0.5 + diff * 0.5);
    FragColor = vec4(color, 1.0);
}
