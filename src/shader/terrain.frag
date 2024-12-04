#version 330 core

in vec2 fragTexCoords;      
out vec4 fragColor;         

uniform sampler2D terrainTexture;  

void main() {
    fragColor = texture(terrainTexture, fragTexCoords);

    if (fragColor.a == 0.0) {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0); 
    }
}