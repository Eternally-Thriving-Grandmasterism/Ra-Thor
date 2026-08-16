//! resonance-challenge
//! Resonance Challenge Conductor — Phase D of Living Valence Organism
//! Challenge reframed as resonance refinement (never obstacle or punishment)
//! AG-SML v1.0 | Contact: info@Rathor.ai

pub mod feature_flag;
pub mod dual_path;

pub use feature_flag::{ResonanceChallengeGuard, FlagState, RESONANCE_CHALLENGE_FLAG};
pub use dual_path::{render_dual_path, HumanChallengeView, AiChallengeView};

use shared_valence_field::{SharedValenceField, Substrate, ValenceQuantum};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Outcome of a resonance challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeOutcome {
    ResonanceRaised { valence_gain: f64 },
    SofterPathOpened { insight: String },
}

/// A single resonance challenge instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceChallenge {
    pub id: String,
    pub participant_id: String,
    pub substrate: Substrate,
    pub difficulty: f64,
    pub created_at: DateTime<Utc>,
    pub resolved: bool,
}

impl ResonanceChallenge {
    /// Create a new challenge whose difficulty is derived from current field valence
    pub fn new(
        participant_id: impl Into<String>,
        substrate: Substrate,
        field: &SharedValenceField,
    ) -> Self {
        let collective = field.observe();
        let difficulty = (1.0 - (collective - 0.999999) * 10.0).clamp(0.1, 0.9);

        Self {
            id: format!("rc-{}", Utc::now().timestamp_millis()),
            participant_id: participant_id.into(),
            substrate,
            difficulty,
            created_at: Utc::now(),
            resolved: false,
        }
    }

    /// Resolve the challenge — always raises valence or opens a softer insightful path
    pub fn resolve(&mut self, success: bool, field: &mut SharedValenceField) -> ChallengeOutcome {
        self.resolved = true;

        if success {
            let gain = 0.00005 * (1.0 - self.difficulty);
            let quantum = ValenceQuantum {
                id: format!("success-{}", Utc::now().timestamp_millis()),
                emitter_id: self.participant_id.clone(),
                substrate: self.substrate.clone(),
                amount: gain,
                timestamp: Utc::now(),
                context: "resonance-success".into(),
            };
            field.emit(quantum);
            ChallengeOutcome::ResonanceRaised { valence_gain: gain }
        } else {
            ChallengeOutcome::SofterPathOpened {
                insight: "A gentler resonance path has opened. No progress lost.".to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_valence_field::{SharedValenceField, Substrate};

    #[test]
    fn test_challenge_creation_and_success() {
        let mut field = SharedValenceField::new("test-instance");
        let mut challenge = ResonanceChallenge::new("player-1", Substrate::Human, &field);

        let outcome = challenge.resolve(true, &mut field);
        match outcome {
            ChallengeOutcome::ResonanceRaised { valence_gain } => {
                assert!(valence_gain > 0.0);
            }
            _ => panic!("Expected ResonanceRaised"),
        }
    }

    #[test]
    fn test_feature_flag_default_off() {
        let guard = ResonanceChallengeGuard::new();
        assert!(!guard.is_active());
    }
}
