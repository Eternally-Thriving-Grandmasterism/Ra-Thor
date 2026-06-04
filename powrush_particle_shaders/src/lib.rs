/*!
# Powrush Particle Shaders — WGSL Cooperative Matrix Support

Investigation of cooperative matrix support in WGSL (as of mid-2026).

## Current Status

As of June 2026, native cooperative matrix support in WGSL is still maturing.
The WebGPU working group has been actively discussing and prototyping cooperative matrix features, but full standardization and broad implementation support are not yet complete.

## Expected / Proposed Features

When support lands, WGSL is expected to include:

- An `enable cooperative_matrix;` directive
- New matrix types or attributes for cooperative matrices
- Built-in functions for cooperative matrix load, multiply-accumulate, and store
- Integration with subgroup/wave execution model

These would map down to the underlying platform's cooperative matrix extension (Vulkan, DX12, Metal).

## Implications for Powrush

Once available, WGSL cooperative matrix support would enable:
- High-performance small neural networks inside compute shaders
- Efficient batch matrix operations on particle data
- Advanced learned culling or procedural effects

Until then, we continue to use well-supported subgroup features (ballot, shuffle, wave-local reductions) for current optimizations.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on expected WGSL cooperative matrix support.
    pub const WGSL_COOPERATIVE_MATRIX_NOTES: &str = r#"
        // As of mid-2026, WGSL cooperative matrix support is in progress.
        // Expected pattern (subject to change):
        //
        // enable cooperative_matrix;
        //
        // let a = cooperative_matrix_load<A_type>(...);
        // let b = cooperative_matrix_load<B_type>(...);
        // let c = cooperative_matrix_multiply_accumulate(a, b, c);
        // cooperative_matrix_store(result, c);
        //
        // Exact syntax and capabilities will be defined by the WGSL spec.
    "#;
}
