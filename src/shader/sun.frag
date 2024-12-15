#version 330 core
in vec3 fragNormal;
in vec3 fragPosition;
out vec4 FragColor;

uniform vec3 lightColor;
uniform float intensity;

// You can simulate a directional light, for example:
uniform vec3 lightDir; // e.g., glm::normalize(vec3(-1.0, -1.0, -1.0))

void main() {
    // Normalize normal
    vec3 N = normalize(fragNormal);

    // Simple Lambertian lighting
    float diff = max(dot(N, -lightDir), 0.0);

    // The sun is very bright; we can tone down directional shading for aesthetics:
    vec3 color = lightColor * (intensity * 0.5 + diff * 0.5);
    FragColor = vec4(color, 1.0);
}
