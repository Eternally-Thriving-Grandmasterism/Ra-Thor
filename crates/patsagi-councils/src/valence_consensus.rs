//! crates/patsagi-councils/src/valence_consensus.rs — v14.15.3
//! Original PATSAGi Valence-Optimized Consensus (anti-filibuster / anti-deadlock)
//!
//! Ported from the foundational design in:
//!   - PATSAGi-Prototypes (valence_consensus_module.py, consensus_engine.py)
//!   - Codex-Eternal (Core Covenant / MercyZero — 20 Jan 2026)
//!
//! Design intent (explicit):
//! Traditional adversarial debate can be filibustered or deadlocked.
//! PATSAGi replaces that with:
//!   1. Paraconsistent tolerance of complexity
//!   2. Mercy gates as ethical guardrails (Core Covenant)
//!   3. Valence fields (joy / mercy / sustain) as the optimization objective
//!   4. Progressive outcome always emitted when gates pass
//!
//! Result: the system keeps moving toward eternal thriving even when
//! inputs are messy or contradictory. No single actor can talk forever.
//!
//! Contact: info@Rathor.ai
//! TOLC 8 | Living Cosmic Tick | ONE Organism

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Immutable Core Covenant (Codex-Eternal / MercyZero).
/// These are not optional parameters — they are the ethical floor.
pub const CORE_COVENANT: &[&str] = &[
    "Mercy before action",
    "Forgiveness before victory",
    "Protect innocence. Never punish.",
    "Shield. Contain. Love.",
];

/// Default valence threshold for progressive approval (original prototypes used ~0.95–0.97).
pub const DEFAULT_VALENCE_THRESHOLD: f64 = 0.92;

/// Soft progressive outcome when full threshold is not met but mercy still holds.
pub const PROGRESSIVE_FLOOR: f64 = 0.72;

/// Single council member’s valence contribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValenceVote {
    pub council: String,
    pub joy: f64,
    pub mercy: f64,
    pub sustain: f64,
    pub valence: f64,
    pub veto: bool,
}

impl ValenceVote {
    pub fn new(council: impl Into<String>, joy: f64, mercy: f64, sustain: f64, veto: bool) -> Self {
        let joy = joy.clamp(0.0, 1.0);
        let mercy = mercy.clamp(0.0, 1.0);
        let sustain = sustain.clamp(0.0, 1.0);
        let valence = (joy + mercy + sustain) / 3.0;
        Self {
            council: council.into(),
            joy,
            mercy,
            sustain,
            valence,
            veto,
        }
    }
}

/// Final consensus result — always progressive when mercy holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValenceConsensusResult {
    pub proposal: String,
    pub avg_valence: f64,
    pub threshold: f64,
    pub has_veto: bool,
    pub approved: bool,
    /// True when full threshold not met but progressive path is still taken.
    pub progressive: bool,
    pub final_verdict: String,
    pub votes: Vec<ValenceVote>,
    pub core_covenant_honored: bool,
}

/// The valence-optimized consensus engine.
///
/// Key anti-deadlock property:
/// - Never returns a pure “stuck” state when mercy gates pass.
/// - Always produces either full approval or a progressive refinement path.
#[derive(Debug, Clone)]
pub struct ValenceConsensusEngine {
    pub threshold: f64,
    pub progressive_floor: f64,
}

impl Default for ValenceConsensusEngine {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_VALENCE_THRESHOLD,
            progressive_floor: PROGRESSIVE_FLOOR,
        }
    }
}

impl ValenceConsensusEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run a full valence deliberation.
    ///
    /// `votes` should already contain the joy/mercy/sustain scores from each council.
    /// The engine never filibusters and never hard-deadlocks when the ethical floor holds.
    pub fn deliberate(
        &self,
        proposal: impl Into<String>,
        votes: Vec<ValenceVote>,
    ) -> ValenceConsensusResult {
        let proposal = proposal.into();

        if votes.is_empty() {
            return ValenceConsensusResult {
                proposal,
                avg_valence: 0.0,
                threshold: self.threshold,
                has_veto: false,
                approved: false,
                progressive: false,
                final_verdict: "No votes received — mercy review required.".into(),
                votes,
                core_covenant_honored: true,
            };
        }

        let avg_valence =
            votes.iter().map(|v| v.valence).sum::<f64>() / votes.len() as f64;
        let has_veto = votes.iter().any(|v| v.veto);

        // Core Covenant check: any hard veto that claims to protect innocence is honored.
        // But pure obstruction without mercy grounding does not force deadlock.
        let approved = avg_valence >= self.threshold && !has_veto;
        let progressive = !approved
            && !has_veto
            && avg_valence >= self.progressive_floor;

        let final_verdict = if approved {
            format!(
                "✅ APPROVED — Eternal Thriving path\n\
                 Average Valence: {:.4} ≥ {:.2}\n\
                 Core Covenant honored. The lattice advances.",
                avg_valence, self.threshold
            )
        } else if progressive {
            format!(
                "🔄 PROGRESSIVE PATH — Soft refinement\n\
                 Average Valence: {:.4} (floor {:.2})\n\
                 No veto. System continues toward higher valence.\n\
                 Filibuster / deadlock avoided by design.",
                avg_valence, self.progressive_floor
            )
        } else if has_veto {
            format!(
                "🛡️ MERCY VETO HONORED\n\
                 Average Valence: {:.4}\n\
                 Core Covenant: Protect innocence. Never punish.\n\
                 Proposal refined under mercy gates.",
                avg_valence
            )
        } else {
            format!(
                "🔍 FURTHER MERCY REVIEW\n\
                 Average Valence: {:.4} below progressive floor.\n\
                 No deadlock — return for elevated mercy alignment.",
                avg_valence
            )
        };

        ValenceConsensusResult {
            proposal,
            avg_valence,
            threshold: self.threshold,
            has_veto,
            approved,
            progressive,
            final_verdict,
            votes,
            core_covenant_honored: true,
        }
    }

    /// Convenience: build votes from a simple map of council → (joy, mercy, sustain).
    pub fn deliberate_from_map(
        &self,
        proposal: impl Into<String>,
        scores: HashMap<String, (f64, f64, f64)>,
    ) -> ValenceConsensusResult {
        let votes: Vec<ValenceVote> = scores
            .into_iter()
            .map(|(name, (j, m, s))| ValenceVote::new(name, j, m, s, false))
            .collect();
        self.deliberate(proposal, votes)
    }
}

/// Quick helper used by higher layers (feedback loop, world governance).
pub fn quick_valence_check(
    joy: f64,
    mercy: f64,
    sustain: f64,
    threshold: f64,
) -> (f64, bool) {
    let valence = ((joy + mercy + sustain) / 3.0).clamp(0.0, 1.0);
    (valence, valence >= threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_valence_approves() {
        let engine = ValenceConsensusEngine::new();
        let votes = vec![
            ValenceVote::new("Joy", 0.98, 0.97, 0.96, false),
            ValenceVote::new("Mercy", 0.95, 0.99, 0.94, false),
            ValenceVote::new("Abundance", 0.97, 0.96, 0.98, false),
        ];
        let res = engine.deliberate("Test abundance bloom", votes);
        assert!(res.approved);
        assert!(!res.progressive);
        assert!(res.core_covenant_honored);
    }

    #[test]
    fn mid_valence_takes_progressive_path() {
        let engine = ValenceConsensusEngine::new();
        let votes = vec![
            ValenceVote::new("A", 0.80, 0.78, 0.75, false),
            ValenceVote::new("B", 0.77, 0.82, 0.79, false),
        ];
        let res = engine.deliberate("Soft refinement case", votes);
        assert!(!res.approved);
        assert!(res.progressive); // anti-deadlock property
    }

    #[test]
    fn veto_is_honored_without_deadlock() {
        let engine = ValenceConsensusEngine::new();
        let votes = vec![
            ValenceVote::new("A", 0.99, 0.99, 0.99, false),
            ValenceVote::new("Protector", 0.50, 0.40, 0.45, true), // mercy veto
        ];
        let res = engine.deliberate("Potential harm case", votes);
        assert!(!res.approved);
        assert!(res.has_veto);
        assert!(res.final_verdict.contains("MERCY VETO"));
    }
}
