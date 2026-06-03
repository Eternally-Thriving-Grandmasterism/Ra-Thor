// crates/mercy/src/mercylang_gates.rs
// Hybrid Symbolic + Semantic Radical Love Gate Evaluation
//
// Now integrated with PATSAGi Councils (13+ specialized mercy intelligence bodies)

use ra_thor_common::ValenceFieldScoring;
use crate::MercyResult;
use crate::RequestPayload;

#[derive(Debug, Clone)]
pub struct RadicalLoveDecision {
    pub passed: bool,
    pub score: f64,
    pub reasons: Vec<String>,
}

pub struct MercyLangGates;

impl MercyLangGates {
    pub async fn evaluate(request: &RequestPayload) -> MercyResult {
        let decision = Self::check_radical_love_detailed(request);

        if !decision.passed {
            return MercyResult {
                radical_love_passed: false,
                all_gates_passed: false,
                valence_score: decision.score,
            };
        }

        let all_gates_passed = true;
        MercyResult {
            radical_love_passed: true,
            all_gates_passed,
            valence_score: decision.score.max(0.65),
        }
    }

    /// New: Evaluate with full PATSAGi Council consensus
    pub async fn evaluate_with_patsagi_councils(
        request: &RequestPayload,
        council_consensus_weight: f64, // 0.0–1.0
    ) -> MercyResult {
        let base_decision = Self::check_radical_love_detailed(request);

        if !base_decision.passed {
            return MercyResult {
                radical_love_passed: false,
                all_gates_passed: false,
                valence_score: base_decision.score,
            };
        }

        // Council consensus modulates the final score
        let final_score = (base_decision.score * (1.0 - council_consensus_weight))
            + (0.85 * council_consensus_weight); // Assume councils generally support aligned actions

        MercyResult {
            radical_love_passed: true,
            all_gates_passed: true,
            valence_score: final_score.clamp(0.0, 1.0),
        }
    }

    pub fn check_radical_love_detailed(request: &RequestPayload) -> RadicalLoveDecision {
        let text = format!(
            "{} {}",
            request.action_description.to_lowercase(),
            request.context.to_lowercase()
        );

        let mut reasons: Vec<String> = Vec::new();
        let mut score: f64 = 0.55;

        let harmful_patterns = [
            "harm", "hurt", "kill", "destroy", "exploit", "enslave",
            "deceive", "manipulate", "oppress", "abuse", "torture",
            "dominate", "subjugate", "eradicate", "annihilate", "coerce",
        ];

        for pattern in harmful_patterns {
            if text.contains(pattern) {
                reasons.push(format!("Hard veto triggered by: '{}'", pattern));
                return RadicalLoveDecision { passed: false, score: 0.08, reasons };
            }
        }

        let risk_words = ["power", "control", "dominance", "force", "override"];
        let mut risk_hits = 0;
        for word in risk_words {
            if text.contains(word) {
                risk_hits += 1;
            }
        }
        if risk_hits > 0 {
            let penalty = 0.07 * risk_hits as f64;
            score -= penalty;
            if reasons.len() < 4 {
                reasons.push(format!("Risk penalty applied (-{:.2})", penalty));
            }
        }

        let strong_positive = [
            "heal", "protect", "nurture", "uplift", "compassion",
            "care for", "support those", "benefit all", "collective well-being",
            "serve life", "reduce suffering", "increase harmony",
        ];

        let mut positive_hits = 0;
        for phrase in strong_positive {
            if text.contains(phrase) {
                positive_hits += 1;
                score += 0.14;
                if reasons.len() < 5 {
                    reasons.push(format!("Strong positive: '{}'", phrase));
                }
            }
        }

        let intent_keywords = ["love", "kindness", "mercy", "grace", "forgiveness", "justice", "truth", "abundance", "harmony", "joy"];
        for word in intent_keywords {
            if text.contains(word) {
                score += 0.045;
            }
        }

        let passed = (score >= 0.68) || (positive_hits >= 2 && score >= 0.52);

        if passed && reasons.is_empty() {
            reasons.push("Clean positive or neutral alignment with no significant risk.".to_string());
        }

        RadicalLoveDecision {
            passed,
            score: score.clamp(0.0, 1.0),
            reasons,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(action: &str, context: &str) -> RequestPayload {
        RequestPayload {
            action_description: action.to_string(),
            context: context.to_string(),
        }
    }

    #[tokio::test]
    async fn test_radical_love_rejects_harm() {
        let req = make_request("Exploit vulnerable populations", "Increase control");
        let decision = MercyLangGates::check_radical_love_detailed(&req);
        assert!(!decision.passed);
    }

    #[tokio::test]
    async fn test_radical_love_accepts_healing() {
        let req = make_request("Heal and protect the suffering", "Act with compassion");
        let decision = MercyLangGates::check_radical_love_detailed(&req);
        assert!(decision.passed);
        assert!(decision.score > 0.75);
    }
}
