// crates/ra-thor-core/src/traits/plasticity.rs
// Ra-Thor™ Plasticity Engine Traits — Absolute Pure Truth Edition
// Multi-timescale, objective-function-free, mercy-gated Hebbian plasticity interface
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::types::{MercyValence, NoveltyScore, BloomIntensity, PlasticityReport};
use serde::{Deserialize, Serialize};
use std::error::Error;

/// Core trait for any component that performs learning / plasticity in the Ra-Thor lattice.
/// All plasticity engines (STDP, BCM, Metaplastic, Synaptic Scaling, ICA, etc.) must implement this.
pub trait PlasticityEngine: Send + Sync {
    /// Process one timestep of input and return a full plasticity report.
    /// This is the main entry point used by `UnifiedSovereignEnergyLatticeCore` and `OpenBCIRaThorBridge`.
    fn process_timestep(
        &mut self,
        input: f64,
        current_valence: MercyValence,
        raw_eeg: Option<&[f64]>,
        dt_ms: f64,
    ) -> Result<PlasticityReport, Box<dyn Error + Send + Sync>>;

    /// Get the current mercy valence of the engine (for monitoring and mercy-gated decisions).
    fn current_valence(&self) -> MercyValence;

    /// Get the current novelty drive (how much intrinsic novelty the engine is generating).
    fn novelty_drive(&self) -> NoveltyScore;

    /// Get the current bloom intensity (how "alive" / thriving the plasticity system feels).
    fn bloom_intensity(&self) -> BloomIntensity;

    /// Optional: Run a self-audit / consistency check (used by FENCA and self-improvement loops).
    fn self_audit(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }

    /// Optional: Reset internal state (useful for testing and sovereign shard resets).
    fn reset(&mut self) {
        // Default no-op
    }
}

/// Trait for any component that can be modulated by mercy valence.
pub trait MercyGated {
    fn set_merry_gating(&mut self, enabled: bool);
    fn is_mercy_gated(&self) -> bool;
}

/// Trait for any component that generates intrinsic novelty (no external reward needed).
pub trait NoveltyProvider {
    fn current_novelty(&self) -> NoveltyScore;
    fn novelty_history(&self) -> &[NoveltyScore];
}

/// Trait for components that participate in the self-improvement / autodidact loop.
pub trait SelfImproving {
    fn propose_improvements(&self) -> Vec<String>;
    fn apply_improvement(&mut self, improvement_name: &str) -> Result<(), Box<dyn Error + Send + Sync>>;
}

/// Combined super-trait for the most powerful plasticity engines (recommended for `RaThorPlasticityEngine`).
pub trait AdvancedPlasticityEngine:
    PlasticityEngine + MercyGated + NoveltyProvider + SelfImproving + Send + Sync
{
}

/// Default implementation helper for `AdvancedPlasticityEngine`.
/// Any struct that implements the four base traits automatically gets this.
impl<T> AdvancedPlasticityEngine for T
where
    T: PlasticityEngine + MercyGated + NoveltyProvider + SelfImproving + Send + Sync,
{
}
