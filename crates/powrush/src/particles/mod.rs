//! crates/powrush/src/particles/mod.rs
//! Resonance Gear Particle System Interface
//!
//! This module defines the data structures and interfaces that will drive
//! Bevy Hanabi (or custom) GPU particle effects for Resonance Gear.

use crate::simulation::ResonanceParticleData;

/// Represents a request to spawn or update particles for a piece of Resonance Gear.
#[derive(Debug, Clone)]
pub struct ResonanceParticleRequest {
    pub data: ResonanceParticleData,
    pub position: nalgebra::Vector3<f32>,
    pub intensity_multiplier: f32, // Can be driven by current harmony or events
}

/// Trait for any particle system backend (Hanabi, custom wgpu, etc.)
pub trait ParticleSystem {
    fn update_resonance_gear(&mut self, request: &ResonanceParticleRequest);
    fn clear(&mut self);
}

// Future: This will be implemented by a BevyHanabiParticleSystem
// that creates and manages EffectAssets based on evolution level.