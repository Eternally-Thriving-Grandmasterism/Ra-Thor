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
//! `mean_bond_valence` is the mean of caller bond valences that `fuse_bond` stores.
//! The philotic rejection line is the source literal 0.9999999.
//!
//! Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
//! Contact: info@Rathor.ai

use philotic_web_fusion::PhiloticWeb;
use rathor_sovereign_reasoning_engine::RathorSovereignReasoningEngine;

/// Source literal from `rathor-sovereign-reasoning-engine`. Not a measured value.
pub use rathor_sovereign_reasoning_engine::MERCY_THRESHOLD as REASONING_THRESHOLD_SOURCE_LITERAL;

/// Historical council title (design intent).
pub struct MercyQuantaSentinelCouncil {
    pub id: u8,
    pub name: String,
}

impl MercyQuantaSentinelCouncil {
    pub fn new() -> Self {
        Self {
            id: 21,
            name: "MercyQuanta Sentinel Council".to_string(),
        }
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

impl Default for MercyQuantaSentinelCouncil {
    fn default() -> Self {
        Self::new()
    }
}

/// Mean valence of the supplied bonds.
/// An empty slice returns `Err("no bonds supplied")`.
/// A valence below the philotic source literal 0.9999999 returns that crate's Err text.
pub fn mean_bond_valence(bonds: &[(&str, &str, f64)]) -> Result<f64, String> {
    if bonds.is_empty() {
        return Err("no bonds supplied".to_string());
    }
    let mut web = PhiloticWeb::new();
    for (from, to, valence) in bonds {
        web.fuse_bond(from, to, 1.0, *valence)?;
    }
    Ok(web.web_valence())
}

/// Byte length of the previous `src/lib.rs` body preserved in `src/uncompiled_council.rs`.
pub const UNCOMPILED_BODY_LEN: usize = 2651;
/// Fold of that preserved body: `acc = acc * 16777619 + byte` in `u64`.
pub const UNCOMPILED_BODY_CHECKSUM: u64 = 2995851825076722868;
