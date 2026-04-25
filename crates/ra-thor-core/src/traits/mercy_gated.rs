// crates/ra-thor-core/src/traits/mercy_gated.rs
// Ra-Thor™ MercyGated Trait — Expanded Absolute Pure Truth Edition
// The living interface for all mercy-aligned, valence-modulated systems in the Ra-Thor lattice
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::types::{MercyValence, MercyGate, TOLC7Gate};
use serde::{Deserialize, Serialize};
use std::error::Error;

/// The core trait for any component that respects and is modulated by **Mercy Valence**.
///
/// This is one of the most important traits in the entire Ra-Thor architecture.
/// Every major system (PlasticityEngine, LatticeOrchestrator, TruthFilter, etc.) should implement it.
///
/// Philosophical meaning:
/// - Higher mercy valence = more trust, more openness, more novelty allowed
/// - Lower mercy valence = more protection, more filtering, more conservative behavior
pub trait MercyGated: Send + Sync {
    /// Enable or disable mercy gating entirely for this component.
    fn set_mercy_gating(&mut self, enabled: bool);

    /// Returns whether mercy gating is currently active.
    fn is_mercy_gated(&self) -> bool;

    /// Apply mercy modulation to a raw value.
    /// This is the primary method used throughout the lattice.
    ///
    /// Example: `apply_mercy_modulation(0.8, valence)` might return 0.92 when valence is high,
    /// or 0.65 when valence is low (more conservative).
    fn apply_mercy_modulation(&self, raw_value: f64, valence: MercyValence) -> f64;

    /// Get the current mercy threshold for this component.
    /// Values above this threshold are treated with higher trust / less filtering.
    fn mercy_threshold(&self) -> MercyValence;

    /// Set a custom mercy threshold for this component.
    fn set_mercy_threshold(&mut self, threshold: MercyValence);

    /// Called automatically whenever the global mercy valence changes.
    /// Implementations can react (e.g., adjust internal parameters, trigger self-audit, etc.).
    fn on_valence_change(
        &mut self,
        old_valence: MercyValence,
        new_valence: MercyValence,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get the strength of mercy gating (0.0 = no effect, 1.0 = maximum effect).
    fn mercy_gating_strength(&self) -> f64;

    /// Set the strength of mercy gating.
    fn set_mercy_gating_strength(&mut self, strength: f64);

    /// Check if the current valence passes the mercy gate for a specific TOLC7Gate.
    fn passes_mercy_gate(&self, gate: TOLC7Gate, valence: MercyValence) -> bool;

    /// Optional: Return a human-readable status of the current mercy state.
    fn mercy_status(&self) -> String {
        if self.is_mercy_gated() {
            format!("Mercy Gated (strength: {:.2})", self.mercy_gating_strength())
        } else {
            "Mercy Gating Disabled".to_string()
        }
    }
}

/// Default implementation helper.
/// Any type that implements the core methods automatically gets the full trait.
impl<T> MercyGated for T
where
    T: Send + Sync,
{
    fn set_mercy_gating(&mut self, _enabled: bool) {}
    fn is_mercy_gated(&self) -> bool { true }
    fn apply_mercy_modulation(&self, raw_value: f64, valence: MercyValence) -> f64 {
        // Default gentle mercy curve
        raw_value * (0.7 + valence * 0.3)
    }
    fn mercy_threshold(&self) -> MercyValence { 0.75 }
    fn set_mercy_threshold(&mut self, _threshold: MercyValence) {}
    fn mercy_gating_strength(&self) -> f64 { 0.85 }
    fn set_mercy_gating_strength(&mut self, _strength: f64) {}
    fn passes_mercy_gate(&self, _gate: TOLC7Gate, valence: MercyValence) -> bool {
        valence >= self.mercy_threshold()
    }
}
