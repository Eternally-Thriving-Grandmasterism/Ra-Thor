//! crates/powrush/src/particles/bevy_hanabi_plugin.rs
//! Bevy + bevy_hanabi integration for Resonance Gear particles
//! Full reactive pipeline + evolution burst effect on level-up.

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
            .add_systems(Startup, setup_resonance_effects)
            .add_systems(Update, (
                spawn_resonance_particles,
                update_resonance_particle_position,
                handle_evolution_changes,
                despawn_evolution_bursts,
            ));
    }
}

/// Resource holding the Hanabi EffectAssets for different evolution levels.
#[derive(Resource, Default)]
pub struct ResonanceParticleAssets {
    pub forge_effects: Vec<Handle<EffectAsset>>,
    pub sanctum_effects: Vec<Handle<EffectAsset>>,
}

/// Marker component for persistent Resonance Gear particle effects.
#[derive(Component, Debug, Clone, Copy)]
pub struct ResonanceGearParticles {
    pub faction: String,
    pub evolution_level: u32,
}

/// Marker + timer for one-time evolution burst effects.
#[derive(Component)]
pub struct EvolutionBurst {
    pub timer: Timer,
}

fn setup_resonance_effects(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut particle_assets: ResMut<ResonanceParticleAssets>,
) {
    for level in 0..=5 {
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
        vec![],
    )
    .with_name(&format!("forge_resonance_lv{}", params.evolution_level))
}

fn create_sanctum_effect_asset(params: &ResonanceEffectParams) -> EffectAsset {
    EffectAsset::new(
        params.particle_count as u32,
        false,
        vec![],
    )
    .with_name(&format!("sanctum_resonance_lv{}", params.evolution_level))
}

fn spawn_resonance_particles(
    mut commands: Commands,
    particle_assets: Res<ResonanceParticleAssets>,
    particle_data: Res<ResonanceParticleData>,
    query: Query<(Entity, &ResonanceGearParticles)>,
) {
    if particle_data.attunement >= 1.0 && particle_data.evolution >= 1 {
        let already_spawned = query.iter().any(|(_, p)| p.faction == "Forge");
        if !already_spawned {
            if let Some(effect_handle) = particle_assets.forge_effects.get(particle_data.evolution as usize) {
                commands.spawn((
                    EffectBundle { effect: effect_handle.clone(), transform: Transform::default(), ..default() },
                    ResonanceGearParticles { faction: "Forge".to_string(), evolution_level: particle_data.evolution },
                ));
                info!("Spawned Forge Resonance Gear particles at evolution level {}", particle_data.evolution);
            }
        }
    }

    if particle_data.attunement >= 2.0 && particle_data.evolution >= 1 {
        let already_spawned = query.iter().any(|(_, p)| p.faction == "Sanctum");
        if !already_spawned {
            if let Some(effect_handle) = particle_assets.sanctum_effects.get(particle_data.evolution as usize) {
                commands.spawn((
                    EffectBundle { effect: effect_handle.clone(), transform: Transform::default(), ..default() },
                    ResonanceGearParticles { faction: "Sanctum".to_string(), evolution_level: particle_data.evolution },
                ));
                info!("Spawned Sanctum Resonance Gear particles at evolution level {}", particle_data.evolution);
            }
        }
    }
}

fn update_resonance_particle_position(
    player_query: Query<&Transform, With<PlayerState>>,
    mut particle_query: Query<(&mut Transform, &ResonanceGearParticles)>,
) {
    if let Ok(player_transform) = player_query.get_single() {
        for (mut particle_transform, _) in &mut particle_query {
            particle_transform.translation = player_transform.translation + Vec3::new(0.0, 1.5, 0.0);
        }
    }
}

fn handle_evolution_changes(
    mut commands: Commands,
    particle_assets: Res<ResonanceParticleAssets>,
    particle_data: Res<ResonanceParticleData>,
    mut particle_query: Query<(Entity, &mut ResonanceGearParticles)>,
) {
    for (entity, mut resonance_particles) in &mut particle_query {
        let current_stored = resonance_particles.evolution_level;
        let player_evolution = particle_data.evolution;

        if player_evolution > current_stored {
            let faction = resonance_particles.faction.clone();

            // Despawn old persistent effect
            commands.entity(entity).despawn();

            // Spawn new higher-evolution persistent effect
            let effects_list = if faction == "Forge" { &particle_assets.forge_effects } else { &particle_assets.sanctum_effects };
            if let Some(new_effect_handle) = effects_list.get(player_evolution as usize) {
                commands.spawn((
                    EffectBundle { effect: new_effect_handle.clone(), transform: Transform::default(), ..default() },
                    ResonanceGearParticles { faction: faction.clone(), evolution_level: player_evolution },
                ));
            }

            // Spawn special one-time evolution burst effect
            spawn_evolution_burst(&mut commands, &particle_assets, &faction, player_evolution);

            info!(
                "{} Resonance Gear evolved from level {} to {} — new particles + burst spawned",
                faction, current_stored, player_evolution
            );
        }
    }
}

fn spawn_evolution_burst(
    commands: &mut Commands,
    particle_assets: &ResonanceParticleAssets,
    faction: &str,
    evolution_level: u32,
) {
    // Use a high-intensity burst effect (reuse highest level asset or create dedicated burst asset in real impl)
    let effects_list = if faction == "Forge" { &particle_assets.forge_effects } else { &particle_assets.sanctum_effects };
    if let Some(burst_handle) = effects_list.get(evolution_level.clamp(0, 5) as usize) {
        commands.spawn((
            EffectBundle {
                effect: burst_handle.clone(),
                transform: Transform::default(),
                ..default()
            },
            EvolutionBurst {
                timer: Timer::from_seconds(0.8, TimerMode::Once),
            },
        ));
    }
}

fn despawn_evolution_bursts(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut EvolutionBurst)>,
) {
    for (entity, mut burst) in &mut query {
        burst.timer.tick(time.delta());
        if burst.timer.just_finished() {
            commands.entity(entity).despawn();
        }
    }
}

// Placeholder PlayerState (use real one from player.rs or simulation in integration)
#[derive(Component)]
struct PlayerState;

// Complete reactive pipeline now includes satisfying evolution burst on level-up.
// Expand Hanabi expressions in create_*_effect_asset for geometric patterns, harmony pulsing, and TOLC influence.