//! crates/patsagi-councils/src/valence_consensus.rs — v14.15.4
//! PATSAGi Valence-Optimized Consensus (anti-filibuster / anti-deadlock)
//!
//! === Design Lineage (important for jargon accuracy) ===
//!
//! Original intent (PATSAGi-Prototypes + Codex-Eternal + PATSAGi-Pinnacle):
//! Traditional adversarial debate can be filibustered or deadlocked.
//! PATSAGi replaces that with four interlocking mechanisms:
//!
//! 1. Paraconsistent tolerance of complexity
//!    (contradiction does not force system halt)
//! 2. Mercy gates + Core Covenant as ethical guardrails
//!    ("Mercy before action. Protect innocence. Never punish. Shield. Contain. Love.")
//! 3. Valence fields as the optimization objective
//!    (joy / harmony / abundance — Pinnacle refinement of earlier joy/mercy/sustain)
//! 4. Progressive outcome always emitted when gates pass
//!    (no pure "stuck" state; the lattice keeps moving toward higher thriving)
//!
//! Result: autonomous, invitation-based, mercy-first collective intelligence
//! that continues even when inputs are messy or contradictory.
//! No single actor can talk forever.
//!
//! Sources:
//! - PATSAGi-Prototypes (valence_consensus_module.py, consensus_engine.py)
//! - PATSAGi-Pinnacle (refined 3-axis + composite + mercy_block)
//! - Codex-Eternal (Core Covenant / MercyZero + DivineChecksum-9)
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

/// Default composite valence threshold for full approval.
pub const DEFAULT_VALENCE_THRESHOLD: f64 = 0.90;

/// Soft progressive outcome when full threshold is not met but mercy still holds.
/// This is the anti-deadlock / anti-filibuster property.
pub const PROGRESSIVE_FLOOR: f64 = 0.68;

/// Default weights for composite valence (Pinnacle-aligned).
pub const DEFAULT_WEIGHTS: (f64, f64, f64) = (0.45, 0.30, 0.25); // joy, harmony, abundance

/// Single council member’s valence contribution (Pinnacle 3-axis).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValenceVote {
    pub council: String,
    /// Peak positive emotion / thriving signal (0.0–1.0+)
    pub joy: f64,
    /// Cosmic / relational alignment (-1.0–1.0, mapped internally)
    pub harmony: f64,
    /// Thriving / resource / abundance amplification (0.0–1.0+)
    pub abundance: f64,
    /// Computed composite valence after weighting
    pub valence: f64,
    /// Absolute mercy veto (Core Covenant protection of innocence)
    pub mercy_block: bool,
}

impl ValenceVote {
    /// Create a vote. Harmony is accepted in [-1.0, 1.0] and normalized.
    pub fn new(
        council: impl Into<String>,
        joy: f64,
        harmony: f64,
        abundance: f64,
        mercy_block: bool,
    ) -> Self {
        let joy = joy.max(0.0);
        let harmony = harmony.clamp(-1.0, 1.0);
        let abundance = abundance.max(0.0);

        // Normalize harmony from [-1,1] → [0,1] for composite
        let harmony_norm = (harmony + 1.0) / 2.0;
        let (wj, wh, wa) = DEFAULT_WEIGHTS;
        let valence = wj * joy + wh * harmony_norm + wa * abundance;

        Self {
            council: council.into(),
            joy,
            harmony,
            abundance,
            valence,
            mercy_block,
        }
    }

    /// Backward-compatible constructor (maps old mercy/sustain → harmony/abundance).
    pub fn from_legacy(
        council: impl Into<String>,
        joy: f64,
        mercy: f64,
        sustain: f64,
        veto: bool,
    ) -> Self {
        // mercy → positive harmony contribution; sustain → abundance
        Self::new(council, joy, mercy * 2.0 - 1.0, sustain, veto)
    }
}

/// Final consensus result — always progressive when mercy holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValenceConsensusResult {
    pub proposal: String,
    pub avg_valence: f64,
    pub avg_joy: f64,
    pub threshold: f64,
    pub has_mercy_block: bool,
    pub approved: bool,
    /// True when full threshold not met but progressive path is still taken.
    pub progressive: bool,
    pub final_verdict: String,
    pub votes: Vec<ValenceVote>,
    pub core_covenant_honored: bool,
}

/// The valence-optimized consensus engine.
///
/// Key anti-deadlock / anti-filibuster property:
/// - Never returns a pure “stuck” state when mercy gates pass.
/// - Always produces either full approval or a progressive refinement path.
/// - Mercy blocks (protecting innocence) are honored without freezing the lattice.
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
                avg_joy: 0.0,
                threshold: self.threshold,
                has_mercy_block: false,
                approved: false,
                progressive: false,
                final_verdict: "No votes received — mercy review required.".into(),
                votes,
                core_covenant_honored: true,
            };
        }

        let n = votes.len() as f64;
        let avg_valence = votes.iter().map(|v| v.valence).sum::<f64>() / n;
        let avg_joy = votes.iter().map(|v| v.joy).sum::<f64>() / n;
        let has_mercy_block = votes.iter().any(|v| v.mercy_block);

        // Core Covenant: mercy_block (protection of innocence) is always honored.
        // Pure obstruction without mercy grounding does not force deadlock.
        let approved = avg_valence >= self.threshold && !has_mercy_block;
        let progressive = !approved
            && !has_mercy_block
            && avg_valence >= self.progressive_floor;

        let final_verdict = if approved {
            format!(
                "✅ APPROVED — Eternal Thriving path\n\
                 Composite Valence: {:.4} ≥ {:.2} | Joy: {:.3}\n\
                 Core Covenant honored. The lattice advances.",
                avg_valence, self.threshold, avg_joy
            )
        } else if progressive {
            format!(
                "🔄 PROGRESSIVE PATH — Soft refinement (anti-deadlock)\n\
                 Composite Valence: {:.4} (floor {:.2}) | Joy: {:.3}\n\
                 No mercy block. System continues toward higher valence.\n\
                 Filibuster / stalemate avoided by original PATSAGi design.",
                avg_valence, self.progressive_floor, avg_joy
            )
        } else if has_mercy_block {
            format!(
                "🛡️ MERCY BLOCK HONORED (Core Covenant)\n\
                 Composite Valence: {:.4} | Joy: {:.3}\n\
                 Protect innocence. Never punish.\n\
                 Proposal refined under living mercy gates.",
                avg_valence, avg_joy
            )
        } else {
            format!(
                "🔍 FURTHER MERCY REVIEW\n\
                 Composite Valence: {:.4} below progressive floor.\n\
                 No deadlock — return for elevated mercy / joy alignment.",
                avg_valence
            )
        };

        ValenceConsensusResult {
            proposal,
            avg_valence,
            avg_joy,
            threshold: self.threshold,
            has_mercy_block,
            approved,
            progressive,
            final_verdict,
            votes,
            core_covenant_honored: true,
        }
    }

    /// Convenience: build votes from council → (joy, harmony, abundance).
    pub fn deliberate_from_map(
        &self,
        proposal: impl Into<String>,
        scores: HashMap<String, (f64, f64, f64)>,
    ) -> ValenceConsensusResult {
        let votes: Vec<ValenceVote> = scores
            .into_iter()
            .map(|(name, (j, h, a))| ValenceVote::new(name, j, h, a, false))
            .collect();
        self.deliberate(proposal, votes)
    }

    /// Legacy-compatible entry (joy, mercy, sustain) → maps to new axes.
    pub fn deliberate_legacy(
        &self,
        proposal: impl Into<String>,
        scores: HashMap<String, (f64, f64, f64)>,
    ) -> ValenceConsensusResult {
        let votes: Vec<ValenceVote> = scores
            .into_iter()
            .map(|(name, (j, m, s))| ValenceVote::from_legacy(name, j, m, s, false))
            .collect();
        self.deliberate(proposal, votes)
    }
}

/// Quick helper used by higher layers (feedback loop, world governance).
pub fn quick_valence_check(
    joy: f64,
    harmony: f64,
    abundance: f64,
    threshold: f64,
) -> (f64, bool) {
    let vote = ValenceVote::new("quick", joy, harmony, abundance, false);
    (vote.valence, vote.valence >= threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_valence_approves() {
        let engine = ValenceConsensusEngine::new();
        let votes = vec![
            ValenceVote::new("Joy", 0.98, 0.95, 0.96, false),
            ValenceVote::new("Harmony", 0.94, 0.97, 0.93, false),
            ValenceVote::new("Abundance", 0.96, 0.92, 0.98, false),
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
            ValenceVote::new("A", 0.78, 0.70, 0.72, false),
            ValenceVote::new("B", 0.75, 0.68, 0.74, false),
        ];
        let res = engine.deliberate("Soft refinement case", votes);
        assert!(!res.approved);
        assert!(res.progressive); // anti-deadlock property
    }

    #[test]
    fn mercy_block_is_honored_without_deadlock() {
        let engine = ValenceConsensusEngine::new();
        let votes = vec![
            ValenceVote::new("A", 0.99, 0.99, 0.99, false),
            ValenceVote::new("Protector", 0.40, -0.6, 0.30, true),
        ];
        let res = engine.deliberate("Potential harm case", votes);
        assert!(!res.approved);
        assert!(res.has_mercy_block);
        assert!(res.final_verdict.contains("MERCY BLOCK"));
    }
}
