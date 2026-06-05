//! GPU Memory Alignment Requirements for Powrush-MMO + Ra-Thor AGI (v17.0 Production)
//!
//! Professional investigation and implementation of correct GPU memory alignment.
//! Critical for stable, high-performance compute shaders when running
//! Powrush-MMO with Ra-Thor AGI and PATSAGi Councils.
//!
//! Key Concepts:
//! - WGSL struct alignment rules
//! - wgpu buffer alignment requirements
//! - Safe CPU <-> GPU data transfer using bytemuck + proper padding
//! - Performance implications of misalignment
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use bytemuck::{Pod, Zeroable};

/// Recommended alignment for storage buffers in most modern GPUs.
pub const STORAGE_BUFFER_ALIGNMENT: u64 = 16;

/// Minimum uniform buffer offset alignment (queried from device at runtime).
pub const MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT: u64 = 256;

/// Epigenetic Profile struct with correct WGSL-compatible alignment.
/// Each field is f32 (4 bytes). We explicitly pad to ensure 16-byte alignment.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuEpigeneticProfile {
    pub volatility: f32,
    pub stability: f32,
    pub ecological_sensitivity: f32,
    pub creative_flow: f32,
    pub mercy_alignment: f32,
    // Padding to reach 16-byte alignment (important for arrays in WGSL)
    pub _padding: [f32; 3],
}

impl GpuEpigeneticProfile {
    pub fn from_cpu(profile: &crate::systems::epigenetic_modulation::EpigeneticProfile) -> Self {
        Self {
            volatility: profile.volatility as f32,
            stability: profile.stability as f32,
            ecological_sensitivity: profile.ecological_sensitivity as f32,
            creative_flow: profile.creative_flow as f32,
            mercy_alignment: profile.mercy_alignment as f32,
            _padding: [0.0; 3],
        }
    }
}

/// Geometric Region struct with proper alignment.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuGeometricRegion {
    pub resonance: f32,
    pub current_layer: u32,
    pub _padding: [f32; 2], // Ensure 16-byte alignment
}

/// Vector embedding type with explicit alignment.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuVector {
    pub data: [f32; 4], // Common size for 4D embeddings (pad as needed)
}

/// Helper to calculate properly aligned buffer size.
pub fn aligned_buffer_size(size: u64, alignment: u64) -> u64 {
    (size + alignment - 1) & !(alignment - 1)
}

/// Create a storage buffer with correct alignment for compute shaders.
pub fn create_aligned_storage_buffer(
    device: &wgpu::Device,
    size: u64,
    label: &str,
) -> wgpu::Buffer {
    let aligned_size = aligned_buffer_size(size, STORAGE_BUFFER_ALIGNMENT);

    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: aligned_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Important notes for Powrush-MMO + Ra-Thor AGI:
///
/// 1. Always use `#[repr(C, align(16))]` for structs passed to WGSL storage buffers.
/// 2. When uploading arrays of profiles/regions, ensure the total buffer size is aligned.
/// 3. For vector embeddings used in similarity search, 16-byte alignment is strongly recommended.
/// 4. Misalignment can cause:
///    - Undefined behavior / crashes on some GPUs
///    - Severe performance penalties
///    - Incorrect simulation results affecting NPC behavior and player experience
///
/// This module ensures professional, stable GPU compute for the entire Powrush ecosystem.
