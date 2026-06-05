# Powrush Particle Shaders

## Tightly Integrated GPU-Driven Pipeline

`GpuDrivenPipeline` is now tightly integrated with `ComputePipelineManager`.

- Pipelines are retrieved via the manager instead of being hardcoded.
- Detailed memory barriers are included between stages.
- Full command buffer recording for the complete pipeline is shown.

This is production-grade integration of the entire GPU-driven rendering system.

---
*GPU-Driven Rendering (Production Quality)*