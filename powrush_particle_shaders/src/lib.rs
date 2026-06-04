/*!
# Powrush Particle Shaders — Visibility Buffer Compression

Optimization of visibility buffer memory usage through bit packing and compact data layouts.

## Optimization Goals

- Reduce memory footprint of the visibility buffer
- Lower memory bandwidth during read/write
- Maintain sufficient precision for shading and depth testing
- Keep the format simple to decode in compute shaders

## Compression Strategy

Instead of storing multiple 32-bit values, we pack data into fewer channels:
- 20 bits for particle instance ID (supports >1 million particles)
- 8 bits for material ID
- 24 bits for depth (stored as uint, sufficient for most cases)
- Packed into two 32-bit values (or one 64-bit if using uint64 storage)

This significantly reduces bandwidth compared to RGBA32.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

/// Compact visibility data packed for efficiency.
/// Total: 52 bits (fits in 64 bits or two u32s).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CompressedParticleVisibility {
    pub packed_data: u32,      // bits 0-19: particle_id, 20-27: material_id
    pub depth_packed: u32,     // 24-bit depth stored as uint
}

impl CompressedParticleVisibility {
    pub fn new(particle_id: u32, material_id: u8, depth: f32) -> Self {
        let id_bits = particle_id & 0xFFFFF;           // 20 bits
        let mat_bits = (material_id as u32) & 0xFF;    // 8 bits
        let packed = id_bits | (mat_bits << 20);

        // Simple float-to-uint depth packing (can be improved with proper quantization)
        let depth_bits = (depth * 16777215.0) as u32; // 24-bit

        Self {
            packed_data: packed,
            depth_packed: depth_bits,
        }
    }

    pub fn particle_id(&self) -> u32 {
        self.packed_data & 0xFFFFF
    }

    pub fn material_id(&self) -> u8 {
        ((self.packed_data >> 20) & 0xFF) as u8
    }

    pub fn depth(&self) -> f32 {
        self.depth_packed as f32 / 16777215.0
    }
}

/// Parameters remain similar but can be extended for compression context.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct VisibilityBufferParams {
    pub view_proj: [[f32; 4]; 4],
    pub screen_size: [u32; 2],
    pub particle_material_id: u32,
}

pub mod compute {
    /// Updated shading pass that works with compressed visibility data.
    pub const COMPRESSED_VISIBILITY_SHADING: &str = r#"
        struct CompressedParticleVisibility {
            packed_data: u32,
            depth_packed: u32,
        };

        struct VisibilityBufferParams {
            view_proj: mat4x4<f32>,
            screen_size: vec2<u32>,
            particle_material_id: u32,
        };

        @group(0) @binding(0) var<uniform> params: VisibilityBufferParams;
        @group(0) @binding(1) var visibility_buffer: texture_storage_2d<rg32uint, read>;
        @group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;

        fn unpack_visibility(data: CompressedParticleVisibility) -> vec3<u32> {
            let particle_id = data.packed_data & 0xFFFFFu;
            let material_id = (data.packed_data >> 20u) & 0xFFu;
            return vec3<u32>(particle_id, material_id, 0u);
        }

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let coord = vec2<i32>(id.xy);
            let data = textureLoad(visibility_buffer, coord, 0);

            let vis = CompressedParticleVisibility {
                packed_data: data.r,
                depth_packed: data.g,
            };

            let unpacked = unpack_visibility(vis);
            let particle_id = unpacked.x;
            let material_id = unpacked.y;

            // TODO: Fetch particle data and shade
            let color = vec4<f32>(f32(material_id) / 255.0, 0.0, 0.0, 1.0);

            textureStore(output_texture, coord, color);
        }
    "#;
}

pub mod wgsl { /* existing shaders */ }
