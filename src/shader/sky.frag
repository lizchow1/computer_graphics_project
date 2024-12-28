#version 330 core

in vec2 vUV;
out vec4 FragColor;

void main()
{
    vec3 horizonColor = vec3(0.9, 0.95, 1.0);  
    vec3 zenithColor  = vec3(0.45, 0.6, 0.9);  

    vec3 finalColor = mix(horizonColor, zenithColor, vUV.y);

    FragColor = vec4(finalColor, 1.0);
}
