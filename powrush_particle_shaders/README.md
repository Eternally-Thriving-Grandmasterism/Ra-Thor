# Powrush Particle Shaders

## Full End-to-End GPU-Driven Pipeline Integration

Added `pipeline.rs` containing a production-grade `GpuDrivenPipeline` example
that wires together all stages:

- Culling (Distance + Hi-Z)
- Compaction
- Visibility Pass
- Shading Pass
- Draw submission via `vkCmdDrawIndirectCount`

Includes proper memory barriers and command buffer recording.

This represents a complete, integrated GPU-driven rendering pipeline.

---
*GPU-Driven Rendering (Production Quality)*