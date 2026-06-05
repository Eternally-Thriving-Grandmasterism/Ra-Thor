//! WGSL Struct Offset Verification for Powrush-MMO (v17.5 Production)
//!
//! Tools to verify that Rust structs match expected WGSL memory layouts.
//! Critical for correctness when passing data between CPU (Bevy ECS) and
//! GPU compute shaders in Powrush-MMO + Ra-Thor AGI.
//!
//! This module provides both documentation of expected offsets and
//! runtime/compile-time verification helpers.
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use bytemuck::{Pod, Zeroable};
use std::mem::{align_of, offset_of, size_of};

/// Verified layout for Epigenetic Profile on GPU.
///
/// Expected WGSL layout (with 16-byte alignment):
/// Offset 0:  volatility            (f32)  size 4
/// Offset 4:  stability             (f32)  size 4
/// Offset 8:  ecological_sensitivity (f32)  size 4
/// Offset 12: creative_flow         (f32)  size 4
/// Offset 16: mercy_alignment       (f32)  size 4
/// Offset 20: _padding              [f32;3] size 12
/// Total size: 32 bytes (multiple of 16)
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

/// Verified layout for Geometric Region.
///
/// Offset 0: resonance      (f32) size 4
/// Offset 4: current_layer  (u32) size 4
/// Offset 8: _padding       [f32;2] size 8
/// Total: 16 bytes
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuGeometricRegion {
    pub resonance: f32,
    pub current_layer: u32,
    pub _padding: [f32; 2],
}

/// Verified 16-byte aligned vector type.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuVector {
    pub data: [f32; 4],
}

/// Compile-time + runtime verification of struct layouts.
/// Call this during development or in tests to ensure layouts haven't drifted.
pub fn verify_gpu_struct_layouts() {
    // Epigenetic Profile verification
    assert_eq!(size_of::<GpuEpigeneticProfile>(), 32, "GpuEpigeneticProfile size mismatch");
    assert_eq!(align_of::<GpuEpigeneticProfile>(), 16, "GpuEpigeneticProfile alignment mismatch");
    assert_eq!(offset_of!(GpuEpigeneticProfile, volatility), 0);
    assert_eq!(offset_of!(GpuEpigeneticProfile, stability), 4);
    assert_eq!(offset_of!(GpuEpigeneticProfile, ecological_sensitivity), 8);
    assert_eq!(offset_of!(GpuEpigeneticProfile, creative_flow), 12);
    assert_eq!(offset_of!(GpuEpigeneticProfile, mercy_alignment), 16);
    assert_eq!(offset_of!(GpuEpigeneticProfile, _padding), 20);

    // Geometric Region verification
    assert_eq!(size_of::<GpuGeometricRegion>(), 16);
    assert_eq!(align_of::<GpuGeometricRegion>(), 16);
    assert_eq!(offset_of!(GpuGeometricRegion, resonance), 0);
    assert_eq!(offset_of!(GpuGeometricRegion, current_layer), 4);
    assert_eq!(offset_of!(GpuGeometricRegion, _padding), 8);

    // Vector verification
    assert_eq!(size_of::<GpuVector>(), 16);
    assert_eq!(align_of::<GpuVector>(), 16);

    println!("[GPU Alignment] All struct layouts verified successfully.");
}

/// Runtime check that can be called at startup.
pub fn assert_gpu_layouts_valid() {
    verify_gpu_struct_layouts();
}
