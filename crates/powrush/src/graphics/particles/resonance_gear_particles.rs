// crates/powrush/src/graphics/particles/resonance_gear_particles.rs
// Resonance Gear Particle System — Powrush RBE
// Priority #4: Epigenetic + Geometric Feedback Loops (Volatility Effects)
// AG-SML v1.0 | TOLC 8 aligned

use bevy::prelude::*;
use bevy_hanabi::prelude::*;
use rand::Rng; // For volatility-based randomness

#[derive(Debug, Clone, Copy)]
pub struct EpigeneticModulation {
    pub strength: f32,
    pub volatility: f32,
    pub layer: u32,
}

impl EpigeneticModulation {
    pub fn new(strength: f32, volatility: f32, layer: u32) -> Self {
        Self { strength: strength.clamp(0.0, 5.0), volatility: volatility.clamp(0.0, 1.0), layer }
    }
    pub fn burst_multiplier(&self) -> f32 { 1.0 + self.strength * 0.35 }
    pub fn accumulation_multiplier(&self) -> f32 { 1.0 + self.strength * 0.2 }
    pub fn visual_intensity(&self) -> f32 { (self.strength * 0.4 + self.volatility * 0.8).clamp(0.8, 3.5) }
    pub fn lifetime_multiplier(&self) -> f32 { (1.0 + self.volatility * 0.6).clamp(0.9, 2.2) }

    pub fn evolution_rate_bonus(&self) -> f32 {
        let base = self.strength * 0.15;
        let layer_bonus = match self.layer {
            0..=1 => 0.05, 2..=3 => 0.12, _ => 0.20,
        };
        (base + layer_bonus).clamp(0.0, 0.8)
    }

    /// Returns a surge multiplier (can be >1 during high volatility moments)
    pub fn volatility_surge_multiplier(&self) -> f32 {
        if self.volatility > 0.25 {
            // Higher volatility layers have chance for surges
            if rand::thread_rng().gen::<f32>() < self.volatility * 0.4 {
                return 1.0 + (self.volatility * 1.5);
            }
        }
        1.0
    }
}

#[derive(Resource, Default, Clone)]
pub struct GeometricResonance {
    pub harmony_score: f32,
    pub current_layer: u32,
    pub resonance_multiplier: f32,
    pub last_updated: f64,
}

impl GeometricResonance {
    pub fn update_from_source(&mut self, harmony: f32, layer: u32) {
        self.harmony_score = harmony.clamp(0.0, 5.0);
        self.current_layer = layer;
        self.resonance_multiplier = 1.0 + (harmony * 0.3);
        self.last_updated = 0.0;
    }
    pub fn is_stale(&self, current_time: f64, max_age: f64) -> bool {
        (current_time - self.last_updated) > max_age
    }
    pub fn layer_modulated_epigenetic_influence(&self) -> EpigeneticModulation {
        let base = self.harmony_score * 0.25;
        let layer_mult = match self.current_layer {
            0 => 1.00, 1 => 1.15, 2 => 1.35, 3 => 1.60, 4 => 1.90, _ => 2.30,
        };
        let volatility = match self.current_layer { 0..=1 => 0.10, 2..=3 => 0.25, _ => 0.40 };
        EpigeneticModulation::new((base * layer_mult).clamp(0.0, 4.5), volatility, self.current_layer)
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GearType {
    #[default] Forge, Sanctum,
}

#[derive(Component)]
pub struct ResonanceGearParticles {
    pub current_evolution: u32,
    pub gear_type: GearType,
    pub epigenetic_accumulation: f32,
}

#[derive(Component)]
pub struct EvolutionBurst { pub lifetime: Timer }

#[derive(Resource)]
pub struct ResonanceEffectAssets {
    pub forge_level_1: Handle<EffectAsset>, pub forge_level_2: Handle<EffectAsset>,
    pub forge_level_3: Handle<EffectAsset>, pub forge_level_4: Handle<EffectAsset>,
    pub sanctum_level_1: Handle<EffectAsset>, pub sanctum_level_2: Handle<EffectAsset>,
    pub sanctum_level_3: Handle<EffectAsset>, pub sanctum_level_4: Handle<EffectAsset>,
    pub evolution_burst: Handle<EffectAsset>,
}

pub struct ResonanceParticlePlugin;

impl Plugin for ResonanceParticlePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ResonanceEffectAssets>()
            .init_resource::<GeometricResonance>()
            .add_systems(Update, (
                spawn_resonance_particles, update_resonance_particle_position,
                handle_evolution_changes, update_evolution_bursts,
                apply_geometric_resonance_to_active_particles, update_geometric_resonance,
                apply_epigenetic_feedback,
            ).chain());
    }
}

fn handle_evolution_changes(
    mut commands: Commands, player_state: Res<PlayerState>, effects: Res<ResonanceEffectAssets>,
    geometric: Res<GeometricResonance>, mut query: Query<(&mut ResonanceGearParticles, Entity)>,
    player_transform_query: Query<&Transform, With<Player>>,
) {
    let Ok(player_transform) = player_transform_query.get_single() else { return };
    for (mut particles, entity) in &mut query {
        if player_state.evolution > particles.current_evolution {
            let old = particles.current_evolution;
            commands.entity(entity).despawn_recursive();
            let new_asset = match (particles.gear_type, player_state.evolution) {
                (GearType::Forge, 1) => &effects.forge_level_1, (GearType::Forge, 2) => &effects.forge_level_2,
                (GearType::Forge, 3) => &effects.forge_level_3, (GearType::Forge, _) => &effects.forge_level_4,
                (GearType::Sanctum, 1) => &effects.sanctum_level_1, (GearType::Sanctum, 2) => &effects.sanctum_level_2,
                (GearType::Sanctum, 3) => &effects.sanctum_level_3, (GearType::Sanctum, _) => &effects.sanctum_level_4,
            };
            let pos = player_transform.translation + Vec3::new(0.0, 1.8, 0.0);
            let modl = geometric.layer_modulated_epigenetic_influence();
            let surge = modl.volatility_surge_multiplier();
            commands.spawn((EffectBundle { effect: new_asset.clone(), transform: Transform::from_translation(pos), ..default() },
                ResonanceGearParticles { current_evolution: player_state.evolution, gear_type: particles.gear_type, epigenetic_accumulation: particles.epigenetic_accumulation }));
            let vis = modl.visual_intensity(); let life = modl.lifetime_multiplier();
            let mut burst = modl.burst_multiplier() * surge;
            apply_geometric_modulation_to_particles(&geometric, &mut burst, &mut 1.0);
            spawn_evolution_burst(&mut commands, &effects, pos, particles.gear_type, player_state.evolution, burst, life, vis);
        }
    }
}

fn spawn_evolution_burst(commands: &mut Commands, effects: &ResonanceEffectAssets, pos: Vec3, gear: GearType, lvl: u32, intensity: f32, life_mult: f32, vis_int: f32) {
    let lt = 1.2 * life_mult * vis_int.clamp(0.8, 2.5);
    commands.spawn((EffectBundle { effect: effects.evolution_burst.clone(), transform: Transform::from_translation(pos), ..default() },
        EvolutionBurst { lifetime: Timer::from_seconds(lt.clamp(0.6, 4.0), TimerMode::Once) }));
}

fn apply_epigenetic_feedback(geometric: Res<GeometricResonance>, mut q: Query<&mut ResonanceGearParticles>) {
    let m = geometric.layer_modulated_epigenetic_influence();
    let surge = m.volatility_surge_multiplier();
    let rate = m.accumulation_multiplier() * surge;
    for mut p in &mut q {
        p.epigenetic_accumulation = (p.epigenetic_accumulation + rate * 0.015).clamp(0.0, 12.0);
    }
}

pub fn apply_geometric_modulation_to_particles(g: &GeometricResonance, burst: &mut f32, mult: &mut f32) {
    *burst = (*burst + g.harmony_score.clamp(0.0,2.0)*0.35 + g.current_layer as f32 *0.08).clamp(0.6,3.5);
    *mult = (*mult + g.harmony_score.clamp(0.0,2.0)*0.45).clamp(0.8,4.5);
}

// PATSAGi note: Added volatility surge system. High-volatility layers can now produce occasional strong epigenetic surges.