# Powrush Particle Shaders

## Visibility Buffer Shading (Compute Pass)

Added a clean, high-quality `VISIBILITY_BUFFER_SHADING` compute shader.

Design goals:
- Minimal but well-structured base
- Uses SoA data access
- Easy to extend with lighting, materials, and effects
- Only shades pixels that have a valid particle in the visibility buffer

This completes the core of the Visibility Buffer + Deferred Shading stage.

---
*GPU-Driven Rendering (Production Quality)*