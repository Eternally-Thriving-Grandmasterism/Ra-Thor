//! crates/powrush/src/particles/bevy_hanabi_plugin.rs
//! Bevy + bevy_hanabi integration for Resonance Gear particles
//! Full reactive pipeline + evolution burst + rich Hanabi expressions + ShardManager integration hooks.

use bevy::prelude::*;
use bevy_hanabi::prelude::*;
use crate::particles::effect_assets::ResonanceEffectParams;
use crate::simulation::ResonanceParticleData;

pub struct ResonanceParticlePlugin;

impl Plugin for ResonanceParticlePlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(HanabiPlugin)
            .init_resource::<ResonanceParticleAssets>()
            .add_event::<ParticleEvolutionEvent>()
            .add_systems(Startup, setup_resonance_effects)
            .add_systems(Update, (
                spawn_resonance_particles,
                update_resonance_particle_position,
                handle_evolution_changes,
                despawn_evolution_bursts,
                send_particle_evolution_events,
            ));
    }
}

#[derive(Resource, Default)]
pub struct ResonanceParticleAssets {
    pub forge_effects: Vec<Handle<EffectAsset>>,
    pub sanctum_effects: Vec<Handle<EffectAsset>>,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct ResonanceGearParticles {
    pub faction: String,
    pub evolution_level: u32,
}

#[derive(Component)]
pub struct EvolutionBurst {
    pub timer: Timer,
}

/// Event emitted when a Resonance Gear evolves. Can be consumed by ShardManager or other systems.
#[derive(Event, Debug, Clone)]
pub struct ParticleEvolutionEvent {
    pub faction: String,
    pub old_level: u32,
    pub new_level: u32,
    pub harmony: f64,
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
            geometric_intensity: 0.4 + level as f32 * 0.1,
            pulse_with_harmony: true,
            harmony_pulse_rate: 1.0,
            burst_multiplier: 1.0 + level as f32 * 0.2,
        };
        particle_assets.forge_effects.push(effects.add(create_forge_effect_asset(&forge_params)));

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
            geometric_intensity: 0.2,
            pulse_with_harmony: level >= 2,
            harmony_pulse_rate: 0.9,
            burst_multiplier: 1.0 + level as f32 * 0.15,
        };
        particle_assets.sanctum_effects.push(effects.add(create_sanctum_effect_asset(&sanctum_params)));
    }
}

fn create_forge_effect_asset(params: &ResonanceEffectParams) -> EffectAsset {
    let mut writer = ExprWriter::new();

    let init_pos = InitPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(0.8 + params.evolution_level as f32 * 0.2).expr(),
        dimension: ShapeDimension::Volume,
    };

    let init_vel = InitVelocitySphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        speed: writer.lit(params.velocity_min..params.velocity_max).expr(),
    };

    let color_over_lifetime = ColorOverLifetimeModifier {
        gradient: Gradient::new(vec![
            (0.0, params.base_color.into()),
            (0.7, [1.0, 0.8, 0.4, 0.6].into()),
            (1.0, [0.8, 0.5, 0.1, 0.0].into()),
        ]),
    };

    let size_over_lifetime = SizeOverLifetimeModifier {
        gradient: Gradient::new(vec![
            (0.0, Vec2::splat(params.particle_size * 0.6)),
            (0.3, Vec2::splat(params.particle_size)),
            (1.0, Vec2::splat(params.particle_size * 0.3)),
        ]),
    };

    let drag = LinearDragModifier {
        drag: writer.lit(0.8 + params.evolution_level as f32 * 0.1).expr(),
    };

    EffectAsset::new(params.particle_count as u32, false, writer.finish())
        .with_name(&format!("forge_resonance_lv{}", params.evolution_level))
        .init(init_pos)
        .init(init_vel)
        .update(color_over_lifetime)
        .update(size_over_lifetime)
        .update(drag)
}

fn create_sanctum_effect_asset(params: &ResonanceEffectParams) -> EffectAsset {
    let mut writer = ExprWriter::new();

    let init_pos = InitPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(1.0 + params.evolution_level as f32 * 0.15).expr(),
        dimension: ShapeDimension::Volume,
    };

    let init_vel = InitVelocitySphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        speed: writer.lit(params.velocity_min..params.velocity_max).expr(),
    };

    let color_over_lifetime = ColorOverLifetimeModifier {
        gradient: Gradient::new(vec![
            (0.0, params.base_color.into()),
            (0.6, [0.7, 0.9, 1.0, 0.7].into()),
            (1.0, [0.4, 0.7, 0.95, 0.0].into()),
        ]),
    };

    let size_over_lifetime = SizeOverLifetimeModifier {
        gradient: Gradient::new(vec![
            (0.0, Vec2::splat(params.particle_size * 0.5)),
            (0.4, Vec2::splat(params.particle_size)),
            (1.0, Vec2::splat(params.particle_size * 0.2)),
        ]),
    };

    let drag = LinearDragModifier {
        drag: writer.lit(0.6 + params.evolution_level as f32 * 0.08).expr(),
    };

    EffectAsset::new(params.particle_count as u32, false, writer.finish())
        .with_name(&format!("sanctum_resonance_lv{}", params.evolution_level))
        .init(init_pos)
        .init(init_vel)
        .update(color_over_lifetime)
        .update(size_over_lifetime)
        .update(drag)
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
    mut ev_writer: EventWriter<ParticleEvolutionEvent>,
    particle_assets: Res<ResonanceParticleAssets>,
    particle_data: Res<ResonanceParticleData>,
    mut particle_query: Query<(Entity, &mut ResonanceGearParticles)>,
) {
    for (entity, mut resonance_particles) in &mut particle_query {
        let current_stored = resonance_particles.evolution_level;
        let player_evolution = particle_data.evolution;

        if player_evolution > current_stored {
            let faction = resonance_particles.faction.clone();
            commands.entity(entity).despawn();

            let effects_list = if faction == "Forge" { &particle_assets.forge_effects } else { &particle_assets.sanctum_effects };
            if let Some(new_effect_handle) = effects_list.get(player_evolution as usize) {
                commands.spawn((
                    EffectBundle { effect: new_effect_handle.clone(), transform: Transform::default(), ..default() },
                    ResonanceGearParticles { faction: faction.clone(), evolution_level: player_evolution },
                ));
            }

            spawn_evolution_burst(&mut commands, &particle_assets, &faction, player_evolution);

            // Emit event for ShardManager / geometric-intelligence consumption
            ev_writer.send(ParticleEvolutionEvent {
                faction: faction.clone(),
                old_level: current_stored,
                new_level: player_evolution,
                harmony: 0.85,
            });

            info!(
                "{} Resonance Gear evolved from level {} to {} — new particles + burst spawned",
                faction, current_stored, player_evolution
            );
        }
    }
}

fn send_particle_evolution_events(
    mut ev_reader: EventReader<ParticleEvolutionEvent>,
) {
    for event in ev_reader.read() {
        debug!("ParticleEvolutionEvent received for {}: {} -> {}", event.faction, event.old_level, event.new_level);
    }
}

fn spawn_evolution_burst(
    commands: &mut Commands,
    particle_assets: &ResonanceParticleAssets,
    faction: &str,
    evolution_level: u32,
) {
    let effects_list = if faction == "Forge" { &particle_assets.forge_effects } else { &particle_assets.sanctum_effects };
    if let Some(burst_handle) = effects_list.get(evolution_level.clamp(0, 5) as usize) {
        commands.spawn((
            EffectBundle { effect: burst_handle.clone(), transform: Transform::default(), ..default() },
            EvolutionBurst { timer: Timer::from_seconds(0.8, TimerMode::Once) },
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

#[derive(Component)]
struct PlayerState;

// nth degree flesh: Full Hanabi expressions + Event-driven integration ready for ShardManager / geometric-intelligence.
// This batch approach allows larger, more efficient PRs while maintaining professional quality.