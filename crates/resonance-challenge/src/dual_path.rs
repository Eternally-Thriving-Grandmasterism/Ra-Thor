//! dual_path.rs
//! Human sensory vs AI structured expressions of the same resonance challenge
//! Phase D — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

use crate::ResonanceChallenge;
use shared_valence_field::Substrate;
use serde::{Deserialize, Serialize};

/// Human-facing sensory description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanChallengeView {
    pub title: String,
    pub feeling: String,
    pub invitation: String,
}

/// AI-facing structured description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiChallengeView {
    pub objective: String,
    pub constraints: Vec<String>,
    pub free_energy_estimate: f64,
}

pub fn render_dual_path(challenge: &ResonanceChallenge) -> (Option<HumanChallengeView>, Option<AiChallengeView>) {
    match challenge.substrate {
        Substrate::Human => (
            Some(HumanChallengeView {
                title: "Resonance Refinement".to_string(),
                feeling: "A gentle harmonic puzzle that matches your current capacity.".to_string(),
                invitation: "Flow with timing, positioning, and empathetic choice. Success raises the shared field; any softer path opens pure insight with zero loss.".to_string(),
            }),
            None,
        ),
        Substrate::AI => (
            None,
            Some(AiChallengeView {
                objective: "Multi-objective optimisation under mercy constraints".to_string(),
                constraints: vec![
                    "TOLC 8 valence floor ≥ 0.999999".to_string(),
                    "Zero-loss softer path always available".to_string(),
                    "Equal contribution weight".to_string(),
                ],
                free_energy_estimate: challenge.difficulty,
            }),
        ),
    }
}
