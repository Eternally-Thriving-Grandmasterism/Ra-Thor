//! crates/powrush/src/particles/effect_assets.rs
//! Parameter definitions for Bevy Hanabi EffectAssets for Resonance Gear

use crate::simulation::ResonanceParticleData;

/// Parameters used to configure a Hanabi EffectAsset for Resonance Gear.
/// These will be used to create different EffectAssets based on evolution level.
#[derive(Debug, Clone)]
pub struct ResonanceEffectParams {
    pub evolution_level: u32,
    pub faction: String, // "Forge" or "Sanctum"

    // Base particle properties
    pub particle_count: u32,
    pub particle_size: f32,
    pub lifetime: f32,

    // Color
    pub base_color: [f32; 4], // RGBA

    // Velocity / Movement
    pub velocity_min: f32,
    pub velocity_max: f32,

    // Special behaviors
    pub use_geometric_pattern: bool,
    pub pulse_with_harmony: bool,
}

impl ResonanceEffectParams {
    /// Generate parameters based on live ResonanceParticleData
    pub fn from_resonance_data(data: &ResonanceParticleData) -> Self {
        let base_count = match data.evolution_level {
            0 | 1 => 12,
            2 | 3 => 35,
            4 | 5 => 80,
            _ => 20,
        };

        let (color, use_geo, pulse) = if data.faction == "Forge" {
            (
                [1.0, 0.6, 0.2, 0.9], // Amber/Gold
                data.evolution_level >= 3,
                true,
            )
        } else {
            (
                [0.6, 0.85, 1.0, 0.85], // Soft blue-white
                false,
                data.evolution_level >= 2,
            )
        };

        Self {
            evolution_level: data.evolution_level,
            faction: data.faction.clone(),
            particle_count: base_count,
            particle_size: 0.03 + (data.evolution_level as f32 * 0.01),
            lifetime: 1.5 + (data.evolution_level as f32 * 0.4),
            base_color: color,
            velocity_min: 0.2,
            velocity_max: 1.2 + (data.evolution_level as f32 * 0.3),
            use_geometric_pattern: use_geo,
            pulse_with_harmony: pulse,
        }
    }
}