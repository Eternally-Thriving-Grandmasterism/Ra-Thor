> **Historical record.** This archived document reflects earlier wording and is not the living claim ceiling. `PUBLIC_CLAIM.lock.md` governs current public claims.

# Powrush Particle Shaders

## Full Descriptor Set Updates

Implemented `update_descriptor_sets()` in `GpuDrivenPipeline`.

This method binds actual resources (buffers and image views) to the
allocated descriptor sets for Culling, Compaction, Visibility, and Shading stages.

The pipeline is now capable of full resource binding and command recording.

---
*GPU-Driven Rendering (Production Quality)*