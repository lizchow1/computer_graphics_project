#version 330 core

in vec3 fragNormal;
out vec4 FragColor;

uniform vec3 lightColor;
uniform float intensity;
uniform vec3 lightDir; 

void main()
{
    float rim = abs(dot(normalize(fragNormal), vec3(0,0,1)));
    rim = 0.5 * rim + 0.5;
    float alpha = clamp(rim, 0.0, 1.0);
    alpha = pow(alpha, 1.5); 
    float brightness = 10.0;
    vec3 color = lightColor * intensity * brightness;
    FragColor = vec4(color, alpha);
}
