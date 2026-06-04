/*!
# Powrush Particle Shaders — Matrix Multiplication Instruction Sets

Exploration of low-level GPU matrix multiplication instruction sets.

## Overview

Modern GPUs provide specialized hardware instructions for matrix multiplication that far outperform traditional scalar or vector FMA (Fused Multiply-Add) operations.

These instructions are the foundation upon which cooperative matrix APIs are built.

## Major Instruction Sets

### NVIDIA (Tensor Cores)
- `mma.sync` family in PTX/SASS
- Shapes such as:
  - m16n8k8, m16n8k16 (f16, tf32)
  - m8n8k4, m8n8k16, m16n8k32 (integer)
- High throughput for mixed-precision computation

### AMD (Matrix Cores)
- MFMA (Matrix Fused Multiply-Add) instructions
- Examples: v_mfma_f32_16x16x16f16, v_mfma_f32_32x32x8f16
- Support for various data types and accumulation precisions

### Intel
- DPAS (Dot Product Accumulate Systolic) instructions
- Used in Xe and Arc architectures for matrix workloads

## Connection to Higher-Level APIs

Cooperative matrix extensions (VK_KHR_cooperative_matrix, etc.) provide a portable abstraction over these low-level instructions.
WGSL will eventually expose similar high-level constructs that map down to the appropriate hardware instructions.

## Relevance to Powrush

Understanding these instruction sets helps anticipate performance characteristics when cooperative matrix features become available for:
- Neural culling / importance scoring
- Advanced procedural effects
- High-throughput batch linear algebra on particle data
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

pub mod compute {
    /// Notes on matrix multiplication instruction sets.
    pub const MATRIX_MUL_INSTRUCTION_NOTES: &str = r#"
        // Low-level matrix multiplication instructions are vendor-specific.
        // Cooperative matrix APIs abstract over them.
        //
        // When WGSL gains cooperative matrix support, the compiler
        // will map high-level operations down to the best available
        // hardware instructions on each platform.
        //
        // For now, we track these capabilities for future optimization.
    "#;
}
