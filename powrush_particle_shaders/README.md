# Powrush Particle Shaders

## Descriptor Set Binding

Added full descriptor set allocation and binding support in `GpuDrivenPipeline`:

- Descriptor pool creation
- Descriptor set allocation from layouts
- Structure ready for updating descriptor sets with actual resources

The pipeline can now properly bind resources for Culling, Compaction, Visibility, and Shading stages.

---
*GPU-Driven Rendering (Production Quality)*