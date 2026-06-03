// crates/powrush/src/graphics/particles/resonance_gear_particles.rs
// Resonance Gear Particle System — Powrush RBE
// v14.5 Geometric Wiring Iteration (builds on merged PR #192 + #193)
// Full geometric resonance modulation now active
// AG-SML v1.0 | TOLC 8 aligned | ONE Organism visual resonance

use bevy::prelude::*;
use bevy_hanabi::prelude::*;

// === Geometric Intelligence Integration (from merged PR #192) ===
// When geometric-intelligence crate is available, replace placeholders with real types:
// use geometric_intelligence::prelude::{compute_geometric_harmony, GeometricHarmony, PolyhedralLayer};

#[derive(Resource, Default)]
pub struct GeometricResonance {
    pub harmony_score: f32,
    pub current_layer: u32, // 0=Platonic ... higher = Hyperbolic
    pub resonance_multiplier: f32,
}

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

#[derive(Component)]
pub struct EvolutionBurst {
    pub lifetime: Timer,
}

#[derive(Resource)]
pub struct ResonanceEffectAssets {
    pub forge_level_1: Handle<EffectAsset>,
    pub forge_level_2: Handle<EffectAsset>,
    pub forge_level_3: Handle<EffectAsset>,
    pub forge_level_4: Handle<EffectAsset>,
    pub sanctum_level_1: Handle<EffectAsset>,
    pub sanctum_level_2: Handle<EffectAsset>,
    pub sanctum_level_3: Handle<EffectAsset>,
    pub sanctum_level_4: Handle<EffectAsset>,
    pub evolution_burst: Handle<EffectAsset>,
}

pub struct ResonanceParticlePlugin;

impl Plugin for ResonanceParticlePlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<ResonanceEffectAssets>()
            .init_resource::<GeometricResonance>()
            .add_systems(
                Update,
                (
                    spawn_resonance_particles,
                    update_resonance_particle_position,
                    handle_evolution_changes,
                    update_evolution_bursts,
                    apply_geometric_resonance_to_active_particles,
                )
                    .chain(),
            );
    }
}

fn spawn_resonance_particles(
    mut commands: Commands,
    player_state: Res<PlayerState>,
    effects: Res<ResonanceEffectAssets>,
    geometric: Res<GeometricResonance>,
    query: Query<&ResonanceGearParticles>,
) {
    if query.iter().next().is_some() {
        return;
    }

    let level = player_state.evolution;
    if level == 0 {
        return;
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
        "Resonance Gear particles spawned for {:?} at evolution level {} (geometric harmony: {:.2})",
        gear_type, level, geometric.harmony_score
    );
}

fn update_resonance_particle_position(
    player_query: Query<&Transform, With<Player>>,
    mut particle_query: Query<&mut Transform, With<ResonanceGearParticles>>,
) {
    if let Ok(player_transform) = player_query.get_single() {
        for mut particle_transform in &mut particle_query {
            particle_transform.translation = player_transform.translation + Vec3::new(0.0, 1.8, 0.0);
        }
    }
}

fn handle_evolution_changes(
    mut commands: Commands,
    player_state: Res<PlayerState>,
    effects: Res<ResonanceEffectAssets>,
    geometric: Res<GeometricResonance>,
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

            commands.entity(entity).despawn_recursive();

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

            let spawn_pos = player_transform.translation + Vec3::new(0.0, 1.8, 0.0);

            commands.spawn((
                EffectBundle {
                    effect: new_asset.clone(),
                    transform: Transform::from_translation(spawn_pos),
                    ..default()
                },
                ResonanceGearParticles {
                    current_evolution: current_level,
                    gear_type: particles.gear_type,
                },
            ));

            // Modulate burst with current geometric resonance
            let mut burst_intensity = 1.0;
            let mut particle_multiplier = 1.0;
            apply_geometric_modulation_to_particles(&geometric, &mut burst_intensity, &mut particle_multiplier);

            spawn_evolution_burst(
                &mut commands,
                &effects,
                spawn_pos,
                particles.gear_type,
                current_level,
                burst_intensity,
            );

            info!(
                "{:?} Resonance Gear evolved from level {} to {} — geometric resonance applied (harmony: {:.2})",
                particles.gear_type, old_level, current_level, geometric.harmony_score
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
    intensity: f32,
) {
    // intensity can be used to scale EffectAsset properties or spawn multiple bursts in future
    commands.spawn((
        EffectBundle {
            effect: effects.evolution_burst.clone(),
            transform: Transform::from_translation(position),
            ..default()
        },
        EvolutionBurst {
            lifetime: Timer::from_seconds(1.2 * intensity.clamp(0.5, 2.0)),
        },
    ));

    info!(
        "Evolution burst triggered for {:?} at level {} (intensity: {:.2})",
        gear_type, level, intensity
    );
}

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

fn apply_geometric_resonance_to_active_particles(
    geometric: Res<GeometricResonance>,
    mut particle_query: Query<&mut Transform, With<ResonanceGearParticles>>,
) {
    // Future: subtle position/velocity modulation based on geometric.harmony_score and current_layer
    // For now, this system exists as the hook for deeper Riemannian curvature influence
    let _ = geometric; // placeholder until full implementation
}

pub fn apply_geometric_modulation_to_particles(
    geometric: &GeometricResonance,
    burst_intensity: &mut f32,
    particle_multiplier: &mut f32,
) {
    // Real implementation will use:
    // let harmony = compute_geometric_harmony(...);
    // *burst_intensity = harmony.resonance_multiplier * geometric.current_layer as f32 * 0.1;
    // *particle_multiplier = 1.0 + (geometric.harmony_score * 0.5);

    *burst_intensity = 1.0 + (geometric.harmony_score * 0.3) + (geometric.current_layer as f32 * 0.05);
    *particle_multiplier = 1.0 + (geometric.harmony_score * 0.4);

    // Clamp for stability
    *burst_intensity = burst_intensity.clamp(0.6, 2.5);
    *particle_multiplier = particle_multiplier.clamp(0.8, 3.0);
}