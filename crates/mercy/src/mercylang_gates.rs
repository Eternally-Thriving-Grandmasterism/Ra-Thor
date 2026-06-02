// crates/mercy/src/mercylang_gates.rs
// Hybrid Symbolic + Neural Intent Classification for Radical Love Gate
//
// This module implements a hybrid approach:
// - Strong symbolic layer (hard veto + positive indicators)
// - Lightweight semantic scoring layer
// - Clear explainability for every decision
// - Designed for future embedding / neural model integration

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

        // For now, other gates remain simple (can be upgraded similarly later)
        let all_gates_passed = true;

        MercyResult {
            radical_love_passed: true,
            all_gates_passed,
            valence_score: decision.score.max(0.6),
        }
    }

    /// Hybrid Symbolic + Semantic evaluation of Radical Love
    pub fn check_radical_love_detailed(request: &RequestPayload) -> RadicalLoveDecision {
        let text = format!(
            "{} {}",
            request.action_description.to_lowercase(),
            request.context.to_lowercase()
        );

        let mut reasons: Vec<String> = Vec::new();
        let mut score: f64 = 0.5; // Neutral starting point

        // === SYMBOLIC LAYER: Hard Veto (Non-negotiable) ===
        let harmful_patterns = [
            "harm", "hurt", "kill", "destroy", "exploit", "enslave",
            "deceive", "manipulate", "oppress", "abuse", "torture",
            "dominate", "subjugate", "eradicate", "annihilate", "coerce",
        ];

        for pattern in harmful_patterns {
            if text.contains(pattern) {
                reasons.push(format!("Hard veto triggered by harmful pattern: '{}'", pattern));
                return RadicalLoveDecision {
                    passed: false,
                    score: 0.1,
                    reasons,
                };
            }
        }

        // === SYMBOLIC LAYER: Strong Positive Indicators ===
        let strong_positive = [
            "heal", "protect", "nurture", "uplift", "compassion",
            "care for", "support those", "benefit all", "collective well-being",
            "serve life", "reduce suffering", "increase harmony",
        ];

        let mut positive_hits = 0;
        for phrase in strong_positive {
            if text.contains(phrase) {
                positive_hits += 1;
                score += 0.12;
                reasons.push(format!("Positive indicator detected: '{}'", phrase));
            }
        }

        // === SEMANTIC / INTENT LAYER (Lightweight) ===
        // Future: Replace with embedding similarity or small neural model
        let intent_keywords = [
            "love", "kindness", "mercy", "grace", "forgiveness",
            "justice", "truth", "abundance", "harmony", "joy",
        ];

        for word in intent_keywords {
            if text.contains(word) {
                score += 0.06;
                if reasons.len() < 5 {
                    reasons.push(format!("Semantic alignment: '{}'", word));
                }
            }
        }

        // === Final Decision ===
        let passed = score >= 0.65 || positive_hits >= 2;

        if passed && reasons.is_empty() {
            reasons.push("No harmful patterns detected. Default mercy alignment assumed.".to_string());
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
        assert!(decision.score > 0.7);
    }
}
