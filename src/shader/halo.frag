#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform vec3  haloColor;  
uniform float haloAlpha;  
uniform float haloIntensity; 

void main()
{
    float dist = length(TexCoord - vec2(0.5));
    float halo = 1.0 - smoothstep(0.0, 0.8, dist);
    vec3 color = haloColor * haloIntensity;
    float alpha = halo * haloAlpha;

    FragColor = vec4(color, alpha);
}
