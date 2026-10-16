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
//!
//! Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
//! Contact: info@Rathor.ai

use patsagi_councils::{ValenceConsensusEngine, ValenceVote};
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
pub struct EternalSovereignSparkCouncil {
    pub id: u8,
    pub name: String,
}

impl EternalSovereignSparkCouncil {
    pub fn new() -> Self {
        Self {
            id: 18,
            name: "Eternal Sovereign Spark Council".to_string(),
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
}

impl Default for EternalSovereignSparkCouncil {
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

/// Byte length of the previous `src/lib.rs` body preserved in `src/uncompiled_council.rs`.
pub const UNCOMPILED_BODY_LEN: usize = 1329;
/// Fold of that preserved body: `acc = acc * 16777619 + byte` in `u64`.
pub const UNCOMPILED_BODY_CHECKSUM: u64 = 10135925569139386985;
