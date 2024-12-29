#version 330 core

in vec2 vUV;  
out vec4 FragColor;

uniform vec3  haloColor;
uniform float haloIntensity; 
uniform float haloAlpha;

void main()
{
    vec2 center = vec2(0.5, 0.5);
    float dist  = distance(vUV, center); 

    float innerRadius = 0.2;
    float outerRadius = 0.5;

    float alpha = 1.0 - smoothstep(innerRadius, outerRadius, dist);

    alpha *= haloAlpha; 

    vec3 finalColor = haloColor * haloIntensity;

    FragColor = vec4(finalColor, alpha);
}
