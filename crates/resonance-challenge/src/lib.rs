//! `ResonanceChallenge::new` sets `difficulty` from `field.observe()` and the source literal `0.999999`.
//! That literal is not a measured value.
//! `resolve(true)` emits `0.00005 * (1.0 - difficulty)`. `resolve(false)` emits no quantum.
//! Contact: info@Rathor.ai

pub mod dual_path;
pub mod feature_flag;

pub use dual_path::{render_dual_path, AiChallengeView, HumanChallengeView};
pub use feature_flag::{FlagState, ResonanceChallengeGuard, RESONANCE_CHALLENGE_FLAG};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use shared_valence_field::{SharedValenceField, Substrate, ValenceQuantum};

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
    /// `difficulty` is `(1.0 - (observe() - 0.999999) * 10.0).clamp(0.1, 0.9)`.
    /// `0.999999` is the source literal stored by `SharedValenceField::new`, not a measured value.
    pub fn new(
        participant_id: impl Into<String>,
        substrate: Substrate,
        field: &SharedValenceField,
    ) -> Self {
        let collective = field.observe();
        // 0.999999 is a source literal, not a measured value.
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

    /// `success` emits one quantum whose amount is the source expression `0.00005 * (1.0 - difficulty)`.
    /// A false `success` sets `resolved` and returns the softer-path string. It does not emit.
    pub fn resolve(&mut self, success: bool, field: &mut SharedValenceField) -> ChallengeOutcome {
        self.resolved = true;

        if success {
            // Source expression. Not a measured gain.
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
    use chrono::Utc;
    use shared_valence_field::{SharedValenceField, Substrate};

    #[test]
    fn fresh_field_clamps_difficulty_to_0_9_and_success_emits_that_gain() {
        let mut field = SharedValenceField::new("test-instance");
        let mut challenge = ResonanceChallenge::new("player-1", Substrate::Human, &field);
        // A new field observes the source literal 0.999999, so the formula clamps to 0.9.
        assert_eq!(challenge.difficulty, 0.9);
        assert_eq!(challenge.participant_id, "player-1");
        assert_eq!(challenge.substrate, Substrate::Human);
        assert_eq!(challenge.resolved, false);

        let outcome = challenge.resolve(true, &mut field);
        let gain = 0.00005 * (1.0 - 0.9);
        match outcome {
            ChallengeOutcome::ResonanceRaised { valence_gain } => {
                assert_eq!(valence_gain, gain);
            }
            ChallengeOutcome::SofterPathOpened { insight } => {
                panic!("expected ResonanceRaised, got {insight}");
            }
        }
        assert_eq!(challenge.resolved, true);
        assert_eq!(field.quanta.len(), 1);
        assert_eq!(field.quanta[0].amount, gain);
        assert_eq!(field.quanta[0].emitter_id, "player-1");
        assert_eq!(field.quanta[0].substrate, Substrate::Human);
        assert_eq!(field.quanta[0].context, "resonance-success");
        // 0.999999 is the source literal in SharedValenceField::new and emit.
        let after = (0.999999_f64 + gain).max(0.999999);
        assert_eq!(field.observe(), after);
    }

    #[test]
    fn failure_emits_nothing_and_returns_the_softer_path_string() {
        let mut field = SharedValenceField::new("test-instance");
        let before = field.observe();
        let mut challenge = ResonanceChallenge::new("player-1", Substrate::AI, &field);
        let outcome = challenge.resolve(false, &mut field);
        assert_eq!(challenge.resolved, true);
        assert_eq!(challenge.substrate, Substrate::AI);
        assert_eq!(field.quanta.len(), 0);
        assert_eq!(field.observe(), before);
        match outcome {
            ChallengeOutcome::SofterPathOpened { insight } => {
                assert_eq!(
                    insight,
                    "A gentler resonance path has opened. No progress lost."
                );
            }
            ChallengeOutcome::ResonanceRaised { valence_gain } => {
                panic!("expected SofterPathOpened, got {valence_gain}");
            }
        }
    }

    #[test]
    fn raised_field_uses_the_difficulty_formula_and_human_view_is_exact() {
        let mut field = SharedValenceField::new("test-instance");
        field.emit(shared_valence_field::ValenceQuantum {
            id: "seed".into(),
            emitter_id: "seed".into(),
            substrate: Substrate::Human,
            amount: 0.02,
            timestamp: Utc::now(),
            context: "seed".into(),
        });
        let observed = field.observe();
        let challenge = ResonanceChallenge::new("player-9", Substrate::Human, &field);
        // 0.999999 is a source literal in the difficulty formula, not a measured value.
        let expect = (1.0 - (observed - 0.999999) * 10.0).clamp(0.1, 0.9);
        assert_eq!(challenge.difficulty, expect);
        assert_ne!(expect, 0.9);

        let (human, ai) = render_dual_path(&challenge);
        assert!(ai.is_none());
        let view = human.expect("human substrate returns a human view");
        assert_eq!(view.title, "Resonance Refinement");
        assert_eq!(
            view.feeling,
            "A gentle harmonic puzzle that matches your current capacity."
        );
        assert_eq!(
            view.invitation,
            "Flow with timing, positioning, and empathetic choice. Success raises the shared field; any softer path opens pure insight with zero loss."
        );
    }

    #[test]
    fn ai_view_copies_difficulty_and_names_the_source_literal() {
        let field = SharedValenceField::new("test-instance");
        let challenge = ResonanceChallenge::new("ai-1", Substrate::AI, &field);
        let (human, ai) = render_dual_path(&challenge);
        assert!(human.is_none());
        let view = ai.expect("ai substrate returns an ai view");
        assert_eq!(
            view.objective,
            "Multi-objective optimisation under mercy constraints"
        );
        assert_eq!(
            view.constraints,
            vec![
                "source literal valence floor 0.999999 (not a measured value)".to_string(),
                "Zero-loss softer path always available".to_string(),
                "Equal contribution weight".to_string(),
            ]
        );
        assert_eq!(view.free_energy_estimate, challenge.difficulty);
        assert_eq!(view.free_energy_estimate, 0.9);
    }

    #[test]
    fn guard_stays_off_unless_both_arguments_are_true() {
        let mut guard = ResonanceChallengeGuard::new();
        assert_eq!(guard.is_active(), false);
        assert_eq!(guard.try_activate(true, false), false);
        assert_eq!(guard.is_active(), false);
        assert_eq!(guard.try_activate(false, true), false);
        assert_eq!(guard.try_activate(true, true), true);
        assert_eq!(guard.is_active(), true);
        guard.deactivate();
        assert_eq!(guard.is_active(), false);
    }
}
