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
//! `council_from_axes` calls `ValenceVote` and `ValenceConsensusEngine`.
//! `DEFAULT_VALENCE_THRESHOLD` is the source literal 0.90.
//! `PROGRESSIVE_FLOOR` is the source literal 0.68.
//! The label follows those inputs, including a mercy-block input.
//! `mean_bond_valence` is the mean of caller bond valences that `fuse_bond` stores.
//! The philotic rejection line is the source literal 0.9999999.
//! `route_persona` returns one historical title from `context` after the engine call.
//! Those titles are design intent.
//!
//! Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
//! Contact: info@Rathor.ai

use patsagi_councils::{ValenceConsensusEngine, ValenceVote};
use philotic_web_fusion::PhiloticWeb;
use rathor_sovereign_reasoning_engine::RathorSovereignReasoningEngine;

/// Source literal from `rathor-sovereign-reasoning-engine`. Not a measured value.
pub use rathor_sovereign_reasoning_engine::MERCY_THRESHOLD as REASONING_THRESHOLD_SOURCE_LITERAL;

/// Source literals from the current patsagi-councils API. Not measured valence.
/// `DEFAULT_VALENCE_THRESHOLD` is 0.90. `PROGRESSIVE_FLOOR` is 0.68.
pub use patsagi_councils::{
    valence_consensus::DEFAULT_WEIGHTS, CouncilGateLabel, DEFAULT_VALENCE_THRESHOLD,
    PROGRESSIVE_FLOOR,
};

/// Historical council title (design intent).
pub struct DynamicPersonaRouterCouncil {
    pub id: u8,
    pub name: String,
    /// Source literal `0.9999999`. Not a measured valence.
    pub valence_threshold: f64,
    /// Historical persona titles (design intent).
    pub active_personas: [&'static str; 7],
}

impl DynamicPersonaRouterCouncil {
    pub fn new() -> Self {
        Self {
            id: 19,
            name: "Dynamic Persona Router Council".to_string(),
            valence_threshold: 0.9999999,
            active_personas: [
                "Eternal Sentinel",
                "Powrush Diplomat",
                "Mercy Gate Auditor",
                "PATSAGi Council Member",
                "Hyperbolic Tiling Visionary",
                "Infinite Horizon Explorer",
                "Eternal Sovereign Spark Guardian",
            ],
        }
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

    /// One-vote deliberation. `valence` and `label` follow the caller axes and `mercy_block`.
    pub fn council_from_axes(
        &self,
        proposal: &str,
        joy: f64,
        harmony: f64,
        abundance: f64,
        mercy_block: bool,
    ) -> CouncilDeliberation {
        let vote = ValenceVote::new(self.name.clone(), joy, harmony, abundance, mercy_block);
        let valence = vote.valence;
        let result = ValenceConsensusEngine::new().deliberate(proposal, vec![vote]);
        let label = result.gate_label();
        CouncilDeliberation {
            proposal: result.proposal,
            valence,
            label,
        }
    }

    /// Returns a historical title chosen from `context`.
    /// Below `REASONING_THRESHOLD_SOURCE_LITERAL`, returns the engine Err text.
    pub fn route_persona(&self, context: &str, proposal_valence: f64) -> Result<String, String> {
        let mut engine = RathorSovereignReasoningEngine::new();
        engine.evaluate_proposal(proposal_valence, false)?;
        let persona = if context.contains("powrush") || context.contains("resource") {
            self.active_personas[1]
        } else if context.contains("foresight") || context.contains("infinite") {
            self.active_personas[5]
        } else if context.contains("mercy") || context.contains("gate") {
            self.active_personas[2]
        } else {
            self.active_personas[0]
        };
        Ok(persona.to_string())
    }
}

impl Default for DynamicPersonaRouterCouncil {
    fn default() -> Self {
        Self::new()
    }
}

/// Deliberation record. `valence` is the weighted composite of the caller axes.
#[derive(Debug, PartialEq)]
pub struct CouncilDeliberation {
    pub proposal: String,
    pub valence: f64,
    pub label: CouncilGateLabel,
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
