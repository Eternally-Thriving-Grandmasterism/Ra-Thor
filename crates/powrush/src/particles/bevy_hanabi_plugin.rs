//! crates/powrush/src/particles/bevy_hanabi_plugin.rs
//! Bevy + bevy_hanabi integration for Resonance Gear particles

use bevy::prelude::*;
use bevy_hanabi::prelude::*;
use crate::particles::effect_assets::ResonanceEffectParams;
use crate::simulation::ResonanceParticleData;

/// Plugin that owns all Resonance Gear particle effects using bevy_hanabi.
pub struct ResonanceParticlePlugin;

impl Plugin for ResonanceParticlePlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(HanabiPlugin)
            .init_resource::<ResonanceParticleAssets>()
            .add_systems(Startup, setup_resonance_effects);
    }
}

/// Resource holding the Hanabi EffectAssets for different evolution levels.
#[derive(Resource, Default)]
pub struct ResonanceParticleAssets {
    pub forge_effects: Vec<Handle<EffectAsset>>,
    pub sanctum_effects: Vec<Handle<EffectAsset>>,
}

fn setup_resonance_effects(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut particle_assets: ResMut<ResonanceParticleAssets>,
) {
    // Pre-create EffectAssets for different evolution levels
    for level in 0..=5 {
        // Forge
        let forge_params = ResonanceEffectParams {
            evolution_level: level,
            faction: "Forge".to_string(),
            particle_count: 20 + level * 15,
            particle_size: 0.04 + level as f32 * 0.015,
            lifetime: 1.8,
            base_color: [1.0, 0.65, 0.2, 0.95],
            velocity_min: 0.3,
            velocity_max: 1.5 + level as f32 * 0.4,
            use_geometric_pattern: level >= 3,
            pulse_with_harmony: true,
        };

        let forge_effect = create_forge_effect_asset(&forge_params);
        particle_assets.forge_effects.push(effects.add(forge_effect));

        // Sanctum
        let sanctum_params = ResonanceEffectParams {
            evolution_level: level,
            faction: "Sanctum".to_string(),
            particle_count: 15 + level * 12,
            particle_size: 0.035 + level as f32 * 0.012,
            lifetime: 2.2,
            base_color: [0.55, 0.82, 1.0, 0.9],
            velocity_min: 0.2,
            velocity_max: 1.0 + level as f32 * 0.35,
            use_geometric_pattern: false,
            pulse_with_harmony: level >= 2,
        };

        let sanctum_effect = create_sanctum_effect_asset(&sanctum_params);
        particle_assets.sanctum_effects.push(effects.add(sanctum_effect));
    }
}

fn create_forge_effect_asset(params: &ResonanceEffectParams) -> EffectAsset {
    EffectAsset::new(
        params.particle_count as u32,
        false,
        vec![], // TODO: Add proper Hanabi expressions / modifiers based on params
    )
    .with_name(&format!("forge_resonance_lv{}", params.evolution_level))
}

fn create_sanctum_effect_asset(params: &ResonanceEffectParams) -> EffectAsset {
    EffectAsset::new(
        params.particle_count as u32,
        false,
        vec![], // TODO: Add proper Hanabi expressions / modifiers based on params
    )
    .with_name(&format!("sanctum_resonance_lv{}", params.evolution_level))
}