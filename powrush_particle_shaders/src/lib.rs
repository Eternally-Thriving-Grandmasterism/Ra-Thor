/*!
# Powrush Particle Shaders — Memory Optimized

Production-grade particle shader logic with explicit GPU memory optimization.

## Memory Optimization Strategy (Professional)
- Compact parameter structs (reduced uniform size)
- Culling helpers to avoid uploading/drawing unnecessary particles
- SoA-friendly layouts for future large-scale particle buffers
- Clear separation between CPU simulation data and GPU upload data
- Comments on best practices for wgpu/Bevy Hanabi memory usage
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

/// Compact GPU-friendly particle shader parameters.
/// Uses f32 where necessary but keeps total size small (~48 bytes).
/// Designed to be uploaded as a single uniform or part of a storage buffer.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct ParticleShaderParams {
    pub base_color: [f32; 3],
    pub intensity: f32,
    pub particle_count: u32,        // Can be used for indirect draw count
    pub lifetime: f32,
    pub velocity_scale: f32,
    pub resonance_field_strength: f32,
    pub faction_hue_shift: f32,
    pub _padding: f32,              // Maintain 16-byte alignment for uniform buffers
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

    /// Returns a reduced particle count for culling / LOD based on distance or importance.
    /// Helps significantly reduce GPU memory bandwidth and draw calls.
    pub fn culled_particle_count(&self, distance: f32, max_distance: f32) -> u32 {
        if distance > max_distance {
            return 0;
        }
        let factor = 1.0 - (distance / max_distance);
        ((self.particle_count as f32) * factor * factor) as u32
    }
}

/// Memory-efficient batch descriptor for uploading many particle systems.
/// Use with storage buffers instead of many small uniform updates.
#[derive(Debug, Clone)]
pub struct ParticleBatch {
    pub params: ParticleShaderParams,
    pub instance_offset: u32,
    pub instance_count: u32,
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

/// Best practices comment (for future Bevy/wgpu integration):
/// - Prefer storage buffers over many uniform buffer updates for large particle counts.
/// - Use the culled_particle_count() for LOD / frustum culling before upload.
/// - Keep ParticleShaderParams under 64 bytes when possible for cache efficiency.
/// - Consider SoA layout (separate position/velocity/color buffers) for very large systems.
