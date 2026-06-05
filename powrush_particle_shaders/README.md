# Powrush Particle Shaders

## Single-Pass Hi-Z Pyramid Generation

Added an advanced single-pass Hi-Z generation shader (`compute::hiz::GENERATE_HIZ_SINGLE_PASS`).

This version uses groupshared memory to generate multiple mip levels of the depth pyramid in a single dispatch, which can be more efficient than multiple dispatches.

This is a more advanced implementation suitable for production GPU-driven occlusion culling.

---
*GPU-Driven Rendering*