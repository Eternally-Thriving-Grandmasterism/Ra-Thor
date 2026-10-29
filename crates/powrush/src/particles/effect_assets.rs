//! crates/powrush/src/particles/effect_assets.rs
//! Parameter definitions for Bevy Hanabi EffectAssets for Resonance Gear

use crate::simulation::ResonanceParticleData;

/// Parameters used to configure a Hanabi EffectAsset for Resonance Gear.
/// Enhanced with fields that can be dynamically influenced by ShardManager / EpigeneticModulation.
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

    // Special behaviors (can be modulated by council blessings)
    pub use_geometric_pattern: bool,
    pub geometric_intensity: f32,      // 0.0 - 1.0+ (higher = more complex sacred geometry patterns)
    pub pulse_with_harmony: bool,
    pub harmony_pulse_rate: f32,       // How strongly harmony affects pulsing
    pub burst_multiplier: f32,         // Multiplier for evolution burst intensity
}

impl ResonanceEffectParams {
    /// Generate rich parameters based on live ResonanceParticleData + optional council influence
    pub fn from_resonance_data(data: &ResonanceParticleData, council_influence: Option<f64>) -> Self {
        let base_count = match data.evolution_level {
            0 | 1 => 12,
            2 | 3 => 40,
            4 | 5 => 90,
            _ => 25,
        };

        let (color, use_geo, pulse) = if data.faction == "Forge" {
            (
                [1.0, 0.6, 0.2, 0.9],
                data.evolution_level >= 3,
                true,
            )
        } else {
            (
                [0.6, 0.85, 1.0, 0.85],
                false,
                data.evolution_level >= 2,
            )
        };

        let influence = council_influence.unwrap_or(0.5) as f32;

        Self {
            evolution_level: data.evolution_level,
            faction: data.faction.clone(),
            particle_count: base_count,
            particle_size: 0.03 + (data.evolution_level as f32 * 0.012),
            lifetime: 1.6 + (data.evolution_level as f32 * 0.35),
            base_color: color,
            velocity_min: 0.2,
            velocity_max: 1.3 + (data.evolution_level as f32 * 0.35),
            use_geometric_pattern: use_geo,
            geometric_intensity: 0.3 + (data.evolution_level as f32 * 0.12) + influence * 0.3,
            pulse_with_harmony: pulse,
            harmony_pulse_rate: 0.8 + influence * 0.6,
            burst_multiplier: 1.0 + (data.evolution_level as f32 * 0.15) + influence * 0.4,
        }
    }
}