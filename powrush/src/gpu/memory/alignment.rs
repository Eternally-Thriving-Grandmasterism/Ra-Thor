//! Deep Investigation: GPU Memory Alignment for Powrush-MMO + Ra-Thor AGI (v17.3 Production)
//!
//! Expanded professional investigation of GPU memory alignment requirements,
//! with practical implementations tailored to Powrush-MMO's multi-pass compute pipeline,
//! vector search, NPC memory, and Ra-Thor AGI integration.
//!
//! Topics Covered:
//! - WGSL alignment rules (including newer @align attribute)
//! - wgpu device limits (minStorageBufferOffsetAlignment, etc.)
//! - Struct padding strategies for performance and correctness
//! - Alignment considerations for compute pipelines and indirect dispatch
//! - Staging buffer alignment when doing async readback
//! - Real-world impact on Powrush-MMO simulation stability and NPC intelligence
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use bytemuck::{Pod, Zeroable};

/// Standard storage buffer alignment (most GPUs).
pub const STORAGE_BUFFER_ALIGNMENT: u64 = 16;

/// Common uniform buffer alignment.
pub const UNIFORM_BUFFER_ALIGNMENT: u64 = 256;

/// Epigenetic Profile - Production aligned struct for GPU compute.
/// Uses explicit padding to guarantee 16-byte alignment required by WGSL storage buffers.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuEpigeneticProfile {
    pub volatility: f32,
    pub stability: f32,
    pub ecological_sensitivity: f32,
    pub creative_flow: f32,
    pub mercy_alignment: f32,
    // 12 bytes padding to reach 16-byte alignment (3 × f32)
    pub _padding: [f32; 3],
}

/// Geometric Region with proper alignment.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuGeometricRegion {
    pub resonance: f32,
    pub current_layer: u32,
    // Padding to maintain 16-byte alignment
    pub _padding: [f32; 2],
}

/// Vector type used for similarity search and NPC memory.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuVector {
    pub data: [f32; 4], // 16 bytes exactly - ideal for many GPUs
}

/// Calculate buffer size aligned to a specific boundary.
pub fn aligned_size(size: u64, alignment: u64) -> u64 {
    (size + alignment - 1) & !(alignment - 1)
}

/// Create a storage buffer with guaranteed alignment.
pub fn create_aligned_storage_buffer(
    device: &wgpu::Device,
    size: u64,
    label: Option<&str>,
) -> wgpu::Buffer {
    let aligned_size = aligned_size(size, STORAGE_BUFFER_ALIGNMENT);

    device.create_buffer(&wgpu::BufferDescriptor {
        label,
        size: aligned_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Important alignment considerations for Powrush-MMO + Ra-Thor AGI:
///
/// 1. **Struct Alignment**: Always use `#[repr(C, align(16))]` for structs in storage buffers.
///    WGSL requires vec3/vec4 and arrays to be 16-byte aligned in many cases.
///
/// 2. **Buffer Copies**: When copying between buffers (e.g. storage → staging for readback),
///    both source and destination must respect alignment requirements.
///
/// 3. **Indirect Dispatch**: The indirect buffer for `dispatch_workgroups_indirect` must be
///    aligned to 4 bytes (but using 16-byte alignment is safer and more consistent).
///
/// 4. **Staging Buffers**: When doing async readback, the staging buffer should use
///    `MAP_READ | COPY_DST`. Its size should be aligned for safety.
///
/// 5. **Performance Impact**: Misaligned accesses can cause:
///    - GPU crashes or undefined behavior on some hardware
///    - Significant performance penalties (cache line splits)
///    - Silent data corruption in complex simulation passes
///
/// 6. **Ra-Thor AGI Impact**: Incorrect alignment in epigenetic or vector data can lead to
///    corrupted NPC memory and poor decision making by PATSAGi Councils.
///
/// Recommendation: Always design GPU data structures with explicit padding from the start.
/// This module provides the foundation for stable, high-performance GPU compute in Powrush-MMO.
