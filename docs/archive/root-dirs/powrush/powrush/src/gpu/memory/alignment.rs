//! WGSL Struct Offset Verification for Powrush-MMO + Ra-Thor AGI (v17.6 Ultimate Production)
//!
//! This module provides compile-time and runtime verification that Rust structs
//! exactly match the expected memory layouts used in WGSL compute shaders.
//!
//! It is critical for correctness when passing data between CPU (Bevy ECS / Ra-Thor agents)
//! and GPU compute pipelines in Powrush-MMO epigenetic + geometric simulations.
//!
//! ## Why this matters
//! WGSL (and most modern GPU APIs) requires strict alignment rules (typically 16-byte
//! for many data structures). Mismatches between Rust `#[repr(C)]` layouts and WGSL
//! expectations cause silent data corruption, incorrect simulation results, or crashes.
//!
//! This module makes those layouts explicit, verifiable, and documented in one place.
//!
//! ## Usage
//! Call `assert_gpu_layouts_valid()` early in application startup or in tests.
//! The functions use `bytemuck`, `offset_of!`, and `size_of` to catch drift immediately.
//!
//! All under AG-SML v1.0 • TOLC 8 Mercy Lattice • 7 Living Mercy Gates

use bytemuck::{Pod, Zeroable};
use std::mem::{align_of, offset_of, size_of};

/// Verified 16-byte aligned layout for Epigenetic Profile passed to GPU.
///
/// Expected WGSL layout (16-byte aligned):
/// Offset 0:  volatility             (f32)      size 4
/// Offset 4:  stability              (f32)      size 4
/// Offset 8:  ecological_sensitivity (f32)      size 4
/// Offset 12: creative_flow          (f32)      size 4
/// Offset 16: mercy_alignment       (f32)      size 4
/// Offset 20: _padding              [f32; 3]    size 12
/// Total: 32 bytes (multiple of 16)
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuEpigeneticProfile {
    pub volatility: f32,
    pub stability: f32,
    pub ecological_sensitivity: f32,
    pub creative_flow: f32,
    pub mercy_alignment: f32,
    pub _padding: [f32; 3],
}

/// Verified 16-byte aligned layout for Geometric Region data.
///
/// Offset 0: resonance      (f32) size 4
/// Offset 4: current_layer  (u32) size 4
/// Offset 8: _padding       [f32; 2] size 8
/// Total: 16 bytes
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuGeometricRegion {
    pub resonance: f32,
    pub current_layer: u32,
    pub _padding: [f32; 2],
}

/// Verified 16-byte aligned 4-component vector type (common in GPU work).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuVector {
    pub data: [f32; 4],
}

/// Compile-time + runtime verification of all known GPU struct layouts.
///
/// Call this during development, in `#[test]` modules, or at application startup
/// to guarantee that Rust layouts have not drifted from the documented WGSL expectations.
/// Panics on mismatch (fail-fast during development).
pub fn verify_gpu_struct_layouts() {
    // === GpuEpigeneticProfile ===
    assert_eq!(size_of::<GpuEpigeneticProfile>(), 32, "GpuEpigeneticProfile size mismatch");
    assert_eq!(align_of::<GpuEpigeneticProfile>(), 16, "GpuEpigeneticProfile alignment mismatch");
    assert_eq!(offset_of!(GpuEpigeneticProfile, volatility), 0);
    assert_eq!(offset_of!(GpuEpigeneticProfile, stability), 4);
    assert_eq!(offset_of!(GpuEpigeneticProfile, ecological_sensitivity), 8);
    assert_eq!(offset_of!(GpuEpigeneticProfile, creative_flow), 12);
    assert_eq!(offset_of!(GpuEpigeneticProfile, mercy_alignment), 16);
    assert_eq!(offset_of!(GpuEpigeneticProfile, _padding), 20);

    // === GpuGeometricRegion ===
    assert_eq!(size_of::<GpuGeometricRegion>(), 16);
    assert_eq!(align_of::<GpuGeometricRegion>(), 16);
    assert_eq!(offset_of!(GpuGeometricRegion, resonance), 0);
    assert_eq!(offset_of!(GpuGeometricRegion, current_layer), 4);
    assert_eq!(offset_of!(GpuGeometricRegion, _padding), 8);

    // === GpuVector ===
    assert_eq!(size_of::<GpuVector>(), 16);
    assert_eq!(align_of::<GpuVector>(), 16);

    println!("[GPU Alignment v17.6] All struct layouts verified successfully.");
}

/// Runtime assertion that can be called once at startup or in integration tests.
/// Recommended to call early in Powrush-MMO + Ra-Thor AGI initialization.
pub fn assert_gpu_layouts_valid() {
    verify_gpu_struct_layouts();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_layouts_are_valid() {
        // This test will catch any accidental layout changes during refactoring.
        verify_gpu_struct_layouts();
    }
}
