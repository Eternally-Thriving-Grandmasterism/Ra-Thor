# Powrush Particle Shaders (Memory Optimized)

## GPU Memory Optimizations Added

- `ParticleShaderParams` is now `#[repr(C)]` + `bytemuck` compatible for direct GPU upload with minimal overhead.
- Added `culled_particle_count(distance, max_distance)` for simple but effective LOD/culling to reduce drawn particles and memory bandwidth.
- `ParticleBatch` struct for efficient batched uploads via storage buffers.
- WGSL snippets remain lightweight.
- Documentation on best practices (storage vs uniform buffers, SoA layouts).

These changes significantly reduce GPU memory pressure when many factions or high-reputation events trigger dense particle effects.

High-reputation factions with strong resonance fields now intelligently use fewer particles when far from camera while maintaining visual quality up close.

## Integration
Use `ParticleShaderParams::culled_particle_count()` before uploading instance data.
Combine with `ReputationSystem` and faction visuals for dynamic quality scaling.

**Memory-efficient visuals that scale with simulation state.**

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*