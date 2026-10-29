//! Council record.
//! The type name and `name` string are a historical title (design intent).
//!
//! `src/uncompiled_council.rs` is not a module.
//! The bytes under its design-only header are the previous `src/lib.rs`.
//!
//! Compiled call: `RathorSovereignReasoningEngine::evaluate_proposal`, then
//! `run_foresight_simulation` when `proposal_valence` is at least the engine
//! source literal `MERCY_THRESHOLD` (0.9999999).
//! The simulation caps are source literals in that engine.
//!
//! Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
//! Contact: info@Rathor.ai

use rathor_sovereign_reasoning_engine::RathorSovereignReasoningEngine;

/// Source literal from `rathor-sovereign-reasoning-engine`. Not a measured value.
pub use rathor_sovereign_reasoning_engine::MERCY_THRESHOLD as REASONING_THRESHOLD_SOURCE_LITERAL;

/// Historical council title (design intent).
pub struct UniversalSovereignAbundanceLatticeCouncil {
    pub id: u8,
    pub name: String,
    /// Source literal `0.9999999`. Not a measured valence.
    valence_threshold: f64,
}

impl UniversalSovereignAbundanceLatticeCouncil {
    pub fn new() -> Self {
        Self {
            id: 28,
            name: "Universal Sovereign Abundance Lattice Council".to_string(),
            valence_threshold: 0.9999999,
        }
    }

    /// Returns the source literal `0.9999999`. Not a measured valence.
    pub fn valence_threshold(&self) -> f64 {
        self.valence_threshold
    }

    /// `valence >= valence_threshold`.
    /// `valence_threshold` is the source literal 0.9999999, not a measured valence.
    pub fn tolc8_mercy_check(&self, valence: f64) -> bool {
        valence >= self.valence_threshold
    }

    /// Below `REASONING_THRESHOLD_SOURCE_LITERAL`, returns the engine Err text.
    /// Otherwise returns `run_foresight_simulation(years)`.
    /// Both caps inside that simulation are source literals.
    /// The engine's second argument stays false. This function does not return the engine Ok text.
    pub fn reasoning_from_inputs(proposal_valence: f64, years: u64) -> Result<f64, String> {
        let mut engine = RathorSovereignReasoningEngine::new();
        engine.evaluate_proposal(proposal_valence, false)?;
        Ok(engine.run_foresight_simulation(years))
    }
}

impl Default for UniversalSovereignAbundanceLatticeCouncil {
    fn default() -> Self {
        Self::new()
    }
}
