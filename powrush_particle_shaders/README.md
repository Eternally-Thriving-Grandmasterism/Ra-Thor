# Powrush Particle Shaders

## Visibility Pass Fragment Shader

Added `visibility::VISIBILITY_PASS`.

This minimal, clean fragment shader is used during the rasterization stage
to write the particle index into the visibility texture. It works together
with the `VISIBILITY_BUFFER_SHADING` compute shader to form the full
Visibility Buffer + Deferred Shading pipeline.

---
*GPU-Driven Rendering (Production Quality)*