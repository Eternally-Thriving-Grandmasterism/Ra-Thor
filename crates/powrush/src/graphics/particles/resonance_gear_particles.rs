// crates/powrush/src/graphics/particles/resonance_gear_particles.rs
// Resonance Gear Particle System — Powrush RBE
// Professional implementation with evolution burst
// AG-SML v1.0 | TOLC 8 aligned | ONE Organism visual resonance

use bevy::prelude::*;
use bevy_hanabi::prelude::*;

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GearType {
    #[default]
    Forge,
    Sanctum,
}

#[derive(Component)]
pub struct ResonanceGearParticles {
    pub current_evolution: u32,
    pub gear_type: GearType,
}

/// One-time intense evolution burst marker (auto-cleanup optional via Hanabi lifetime)
#[derive(Component)]
pub struct EvolutionBurst {
    pub lifetime: Timer,
}

#[derive(Resource)]
pub struct ResonanceEffectAssets {
    // Pre-built Hanabi EffectAssets (create these in your asset loading system)
    pub forge_level_1: Handle<EffectAsset>,
    pub forge_level_2: Handle<EffectAsset>,
    pub forge_level_3: Handle<EffectAsset>,
    pub forge_level_4: Handle<EffectAsset>,
    pub sanctum_level_1: Handle<EffectAsset>,
    pub sanctum_level_2: Handle<EffectAsset>,
    pub sanctum_level_3: Handle<EffectAsset>,
    pub sanctum_level_4: Handle<EffectAsset>,

    /// Special intense short-lived burst effect (high energy, radial explosion)
    pub evolution_burst: Handle<EffectAsset>,
}

pub struct ResonanceParticlePlugin;

impl Plugin for ResonanceParticlePlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<ResonanceEffectAssets>()
            .add_systems(
                Update,
                (
                    spawn_resonance_particles,
                    update_resonance_particle_position,
                    handle_evolution_changes,
                    update_evolution_bursts, // optional cleanup if not using Hanabi kill-on-age
                )
                    .chain(),
            );
    }
}

fn spawn_resonance_particles(
    mut commands: Commands,
    player_state: Res<PlayerState>, // Your resource with evolution: u32 + attunement
    effects: Res<ResonanceEffectAssets>,
    query: Query<&ResonanceGearParticles>,
) {
    // Prevent duplicate spawning
    if query.iter().next().is_some() {
        return;
    }

    let level = player_state.evolution;
    if level == 0 {
        return; // Not yet attuned
    }

    let (asset, gear_type) = match player_state.gear_attunement {
        GearType::Forge => match level {
            1 => (&effects.forge_level_1, GearType::Forge),
            2 => (&effects.forge_level_2, GearType::Forge),
            3 => (&effects.forge_level_3, GearType::Forge),
            _ => (&effects.forge_level_4, GearType::Forge),
        },
        GearType::Sanctum => match level {
            1 => (&effects.sanctum_level_1, GearType::Sanctum),
            2 => (&effects.sanctum_level_2, GearType::Sanctum),
            3 => (&effects.sanctum_level_3, GearType::Sanctum),
            _ => (&effects.sanctum_level_4, GearType::Sanctum),
        },
    };

    commands.spawn((
        EffectBundle {
            effect: asset.clone(),
            transform: Transform::default(),
            ..default()
        },
        ResonanceGearParticles {
            current_evolution: level,
            gear_type,
        },
    ));

    info!(
        "Resonance Gear particles spawned for {:?} at evolution level {}",
        gear_type, level
    );
}

fn update_resonance_particle_position(
    player_query: Query<&Transform, With<Player>>, // Your player entity
    mut particle_query: Query<&mut Transform, With<ResonanceGearParticles>>,
) {
    if let Ok(player_transform) = player_query.get_single() {
        for mut particle_transform in &mut particle_query {
            // Keep particles attached with a nice offset (e.g. above player or around gear)
            particle_transform.translation = player_transform.translation + Vec3::new(0.0, 1.8, 0.0);
        }
    }
}

fn handle_evolution_changes(
    mut commands: Commands,
    player_state: Res<PlayerState>,
    effects: Res<ResonanceEffectAssets>,
    mut query: Query<(&mut ResonanceGearParticles, Entity)>,
    player_transform_query: Query<&Transform, With<Player>>,
) {
    let Ok(player_transform) = player_transform_query.get_single() else {
        return;
    };

    for (mut particles, entity) in &mut query {
        let current_level = player_state.evolution;
        if current_level > particles.current_evolution {
            let old_level = particles.current_evolution;

            // Despawn old effect
            commands.entity(entity).despawn_recursive();

            // Spawn new higher-evolution effect
            let new_asset = match (particles.gear_type, current_level) {
                (GearType::Forge, 1) => &effects.forge_level_1,
                (GearType::Forge, 2) => &effects.forge_level_2,
                (GearType::Forge, 3) => &effects.forge_level_3,
                (GearType::Forge, _) => &effects.forge_level_4,
                (GearType::Sanctum, 1) => &effects.sanctum_level_1,
                (GearType::Sanctum, 2) => &effects.sanctum_level_2,
                (GearType::Sanctum, 3) => &effects.sanctum_level_3,
                (GearType::Sanctum, _) => &effects.sanctum_level_4,
            };

            commands.spawn((
                EffectBundle {
                    effect: new_asset.clone(),
                    transform: Transform::from_translation(player_transform.translation + Vec3::new(0.0, 1.8, 0.0)),
                    ..default()
                },
                ResonanceGearParticles {
                    current_evolution: current_level,
                    gear_type: particles.gear_type,
                },
            ));

            // === EVOLUTION BURST ===
            spawn_evolution_burst(
                &mut commands,
                &effects,
                player_transform.translation + Vec3::new(0.0, 1.8, 0.0),
                particles.gear_type,
                current_level,
            );

            info!(
                "{:?} Resonance Gear evolved from level {} to {} — new particles + burst spawned",
                particles.gear_type, old_level, current_level
            );
        }
    }
}

fn spawn_evolution_burst(
    commands: &mut Commands,
    effects: &ResonanceEffectAssets,
    position: Vec3,
    gear_type: GearType,
    level: u32,
) {
    // Spawn intense, short-lived burst effect
    commands.spawn((
        EffectBundle {
            effect: effects.evolution_burst.clone(),
            transform: Transform::from_translation(position),
            ..default()
        },
        EvolutionBurst {
            lifetime: Timer::from_seconds(1.2, TimerMode::Once),
        },
    ));

    info!(
        "Evolution burst triggered for {:?} at level {} (intense resonance release)",
        gear_type, level
    );
}

/// Optional cleanup system (use only if your burst EffectAsset does not self-kill via age/lifetime)
fn update_evolution_bursts(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut EvolutionBurst)>,
) {
    for (entity, mut burst) in &mut query {
        burst.lifetime.tick(time.delta());
        if burst.lifetime.just_finished() {
            commands.entity(entity).despawn();
        }
    }
}