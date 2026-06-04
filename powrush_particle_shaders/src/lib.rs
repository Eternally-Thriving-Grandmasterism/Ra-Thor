/*!
# Powrush Particle Shaders — Compute Shader Culling

Production-grade particle system with GPU compute shader culling.

## Why Compute Shader Culling?
CPU-side culling (`culled_particle_count`) is good for simple cases.
For high particle counts (hundreds of thousands across many factions/events),
GPU compute culling is significantly more efficient:
- Parallel culling across thousands of particles
- Can write compact visible index buffers for indirect drawing
- Reduces CPU → GPU upload bandwidth
- Enables more advanced culling (frustum, occlusion hints, importance based on reputation)
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticleShaderParams {
    pub base_color: [f32; 3],
    pub intensity: f32,
    pub particle_count: u32,
    pub lifetime: f32,
    pub velocity_scale: f32,
    pub resonance_field_strength: f32,
    pub faction_hue_shift: f32,
    pub _padding: f32,
}

impl Default for ParticleShaderParams {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            particle_count: 64,
            lifetime: 1.0,
            velocity_scale: 1.0,
            resonance_field_strength: 0.0,
            faction_hue_shift: 0.0,
            _padding: 0.0,
        }
    }
}

impl ParticleShaderParams {
    pub fn from_particle_params(
        faction: Faction,
        visual: &FactionVisualIdentity,
        particle: &ParticleParams,
        reputation: f64,
        harmony: f64,
        council_valence_bonus: f64,
    ) -> Self {
        let intensity = particle.intensity * (0.8 + reputation as f32 * 0.3);
        let resonance = ((harmony as f32 - 0.5) * 0.6 + council_valence_bonus as f32 * 0.4).clamp(-0.3, 0.8);

        Self {
            base_color: visual.particle_color,
            intensity,
            particle_count: particle.base_count,
            lifetime: particle.lifetime_multiplier,
            velocity_scale: particle.velocity_scale,
            resonance_field_strength: resonance,
            faction_hue_shift: match faction {
                Faction::Forge => 0.02,
                Faction::Evolutionary => -0.04,
                Faction::Harmony => 0.06,
            },
            _padding: 0.0,
        }
    }

    pub fn culled_particle_count(&self, distance: f32, max_distance: f32) -> u32 {
        if distance > max_distance {
            return 0;
        }
        let factor = 1.0 - (distance / max_distance);
        ((self.particle_count as f32) * factor * factor) as u32
    }
}

/// Parameters for GPU compute shader culling pass.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ComputeCullingParams {
    pub camera_position: [f32; 3],
    pub max_cull_distance: f32,
    pub importance_threshold: f32,      // e.g. based on reputation/harmony
    pub total_particles: u32,
    pub _padding: [f32; 3],
}

pub mod compute {
    /// WGSL compute shader for distance + importance culling.
    /// Dispatched with one thread per particle (or workgroup of 64/256).
    /// Writes visible particle indices into an output buffer for indirect drawing.
    pub const CULLING_SHADER: &str = r#"
        struct ComputeCullingParams {
            camera_position: vec3<f32>,
            max_cull_distance: f32,
            importance_threshold: f32,
            total_particles: u32,
        }

        @group(0) @binding(0) var<uniform> params: ComputeCullingParams;
        @group(0) @binding(1) var<storage, read> particle_positions: array<vec3<f32>>;
        @group(0) @binding(2) var<storage, read_write> visible_indices: array<u32>;
        @group(0) @binding(3) var<storage, read_write> visible_count: atomic<u32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= params.total_particles) {
                return;
            }

            let pos = particle_positions[index];
            let dist = distance(pos, params.camera_position);

            // Simple distance culling + importance threshold
            if (dist < params.max_cull_distance) {
                let importance = 1.0; // Can be extended with reputation/harmony per particle
                if (importance >= params.importance_threshold) {
                    let slot = atomicAdd(&visible_count, 1u);
                    visible_indices[slot] = index;
                }
            }
        }
    "#;
}

pub mod wgsl {
    pub const BURST_RESONANCE: &str = r#"
        fn powrush_particle_burst(
            position: vec3<f32>,
            velocity: vec3<f32>,
            age: f32,
            intensity: f32,
            resonance: f32,
            hue_shift: f32,
        ) -> vec4<f32> {
            let t = age * 3.14159;
            let resonance_wave = sin(t * 2.0 + resonance * 6.28) * 0.5 + 0.5;
            var color = base_color;
            color.r = clamp(color.r + hue_shift, 0.0, 1.0);
            let final_intensity = intensity * (0.6 + resonance_wave * 0.4);
            let alpha = (1.0 - age) * final_intensity;
            return vec4<f32>(color, alpha);
        }
    "#;

    pub const RESONANCE_TRAIL: &str = r#"
        fn powrush_resonance_trail(
            position: vec3<f32>,
            velocity: vec3<f32>,
            age: f32,
            resonance_strength: f32,
        ) -> vec4<f32> {
            let wave = sin(age * 8.0 + resonance_strength * 12.0) * 0.5 + 0.5;
            let alpha = (1.0 - age * 0.7) * wave * resonance_strength;
            return vec4<f32>(base_color * 1.2, alpha);
        }
    "#;
}

pub fn get_resonance_effect(params: &ParticleShaderParams) -> &'static str {
    if params.resonance_field_strength > 0.4 {
        wgsl::RESONANCE_TRAIL
    } else {
        wgsl::BURST_RESONANCE
    }
}
