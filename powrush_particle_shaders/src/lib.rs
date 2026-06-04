/*!
# Powrush Particle Shaders — Cooperative Matrix Formats

Exploration of matrix formats used in cooperative matrix extensions.

## What are Cooperative Matrix Formats?

In cooperative matrix APIs (e.g. VK_KHR_cooperative_matrix), matrices are not arbitrary. They have specific:

- **Element types** (f16, f32, s8, u8, s32, etc.)
- **Matrix shapes / fragment sizes** (e.g. 16x16, 8x8, 16x8, 32x8)
- **Storage layouts** (row-major, column-major, or hardware-optimized tiled layouts)

These formats are chosen to match the underlying hardware matrix units (Tensor Cores, Matrix Cores) for maximum efficiency.

Threads in a cooperative group work together to load, compute on, and store these matrix fragments.

## Why Formats Matter

- Different formats have different performance characteristics and precision.
- Some formats support accumulation in higher precision (e.g., f16 multiply → f32 accumulate).
- Layout affects how efficiently data can be loaded from memory into the cooperative matrix.
- Compatibility varies across GPU vendors and generations.

## Relevance to Powrush

As cooperative matrix support matures in WGSL, understanding formats will be important for:
- Implementing small neural networks for culling or visual effects
- Efficient batch transformations of particle data
- Future high-performance procedural effects

The choice of format will impact both performance and numerical behavior.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on cooperative matrix formats for future use.
    pub const COOPERATIVE_MATRIX_FORMAT_NOTES: &str = r#"
        // Common cooperative matrix configurations (as of 2026):
        // - A: f16, 16x16 or 8x8
        // - B: f16, matching A
        // - C/D (accumulator): f16 or f32
        //
        // Loading typically uses cooperative_matrix_load with specific layouts.
        // Computation uses cooperative_matrix_multiply_accumulate.
        //
        // Exact shapes and types depend on the GPU and extension version.
        // Future WGSL will likely expose these as parameterized types.
    "#;
}
