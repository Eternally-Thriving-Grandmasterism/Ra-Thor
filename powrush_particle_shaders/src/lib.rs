/*!
# Powrush Particle Shaders

Production-grade particle shader logic for Powrush Resonance Gear visuals.

## Purpose
Provides the shader-ready parameter system and WGSL logic snippets that consume
`FactionVisualIdentity` + `ParticleParams` + reputation/harmony state.

This is the bridge between simulation (reputation, council, RBE) and rendering.
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

/// Final shader uniforms ready to be uploaded to GPU (wgpu/Bevy)
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ParticleShaderParams {
    pub base_color: [f32; 3],
    pub intensity: f32,
    pub particle_count: u32,
    pub lifetime: f32,
    pub velocity_scale: f32,
    pub resonance_field_strength: f32, // modulated by harmony + council valence
    pub faction_hue_shift: f32,
}

impl ParticleShaderParams {
    /// Creates final shader params from faction visual identity + dynamic particle params
    /// Reputation and harmony provide the "life" in the visuals.
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
        }
    }
}

/// WGSL shader logic snippets (production ready for Bevy Hanabi or custom wgpu pipeline)
pub mod wgsl {
    /// Base burst + resonance field shader (can be composed into Hanabi effects)
    pub const BURST_RESONANCE: &str = r#"
        // Powrush Resonance Gear - Burst + Resonance Field
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

            // Dynamic color with faction hue shift
            var color = base_color;
            color.r = clamp(color.r + hue_shift, 0.0, 1.0);

            // Intensity + resonance modulation
            let final_intensity = intensity * (0.6 + resonance_wave * 0.4);
            let alpha = (1.0 - age) * final_intensity;

            return vec4<f32>(color, alpha);
        }
    "#;

    /// Trail / resonance field effect (for high-harmony or high-reputation states)
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

/// Helper to get a complete shader effect based on current state
pub fn get_resonance_effect(
    params: &ParticleShaderParams,
) -> &'static str {
    if params.resonance_field_strength > 0.4 {
        wgsl::RESONANCE_TRAIL
    } else {
        wgsl::BURST_RESONANCE
    }
}
