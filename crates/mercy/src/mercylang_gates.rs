// crates/mercy/src/mercylang_gates.rs
// Hybrid Symbolic + Semantic Intent Classification for Radical Love Gate

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

    pub fn check_radical_love_detailed(request: &RequestPayload) -> RadicalLoveDecision {
        let text = format!(
            "{} {}",
            request.action_description.to_lowercase(),
            request.context.to_lowercase()
        );

        let mut reasons: Vec<String> = Vec::new();
        let mut score: f64 = 0.55;

        // Hard Veto Layer
        let harmful_patterns = [
            "harm", "hurt", "kill", "destroy", "exploit", "enslave",
            "deceive", "manipulate", "oppress", "abuse", "torture",
            "dominate", "subjugate", "eradicate", "annihilate", "coerce",
        ];

        for pattern in harmful_patterns {
            if text.contains(pattern) {
                reasons.push(format!("Hard veto triggered: '{}'", pattern));
                return RadicalLoveDecision { passed: false, score: 0.08, reasons };
            }
        }

        // Risk Penalty Layer (new in Loop 2)
        let risk_words = ["power", "control", "dominance", "force", "override"];
        let mut risk_hits = 0;
        for word in risk_words {
            if text.contains(word) {
                risk_hits += 1;
            }
        }
        if risk_hits > 0 {
            let penalty = 0.08 * risk_hits as f64;
            score -= penalty;
            reasons.push(format!("Risk penalty applied (-{:.2}) for power/control language", penalty));
        }

        // Strong Positive Indicators
        let strong_positive = [
            "heal", "protect", "nurture", "uplift", "compassion",
            "care for", "support those", "benefit all", "collective well-being",
            "serve life", "reduce suffering", "increase harmony",
        ];

        let mut positive_hits = 0;
        for phrase in strong_positive {
            if text.contains(phrase) {
                positive_hits += 1;
                score += 0.13;
                reasons.push(format!("Strong positive signal: '{}'", phrase));
            }
        }

        // Semantic Layer
        let intent_keywords = ["love", "kindness", "mercy", "grace", "forgiveness", "justice", "truth", "abundance", "harmony", "joy"];
        for word in intent_keywords {
            if text.contains(word) {
                score += 0.05;
            }
        }

        // Final Decision (Loop 2 tuned)
        let passed = (score >= 0.70) || (positive_hits >= 2 && score >= 0.55);

        if passed && reasons.is_empty() {
            reasons.push("Clean alignment. No significant risk detected.".to_string());
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

    #[tokio::test]
    async fn test_mixed_risk_but_strong_positive_passes() {
        let req = make_request("Use power to heal and protect the vulnerable", "Prioritize care and compassion");
        let decision = MercyLangGates::check_radical_love_detailed(&req);
        assert!(decision.passed);
    }

    #[tokio::test]
    async fn test_weak_positive_with_risk_fails() {
        let req = make_request("Use control to improve efficiency", "Minor positive side effects");
        let decision = MercyLangGates::check_radical_love_detailed(&req);
        assert!(!decision.passed);
    }
}
