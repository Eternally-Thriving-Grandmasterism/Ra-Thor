/*!
# Tiered Mercy Evaluator

Implements a three-tier evaluation system for TOLC principles.
*/

use crate::mercylang_gates::{MercyLangGates, MercyResult};
use crate::RequestPayload;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MercyTier {
    Foundational,
    Expanded,
    Cosmic,
}

#[derive(Debug, Clone)]
pub struct TieredMercyResult {
    pub tier: MercyTier,
    pub passed: bool,
    pub foundational_passed: bool,
    pub score: f64,
    pub reasons: Vec<String>,
}

pub struct TieredMercyEvaluator;

impl TieredMercyEvaluator {
    pub async fn evaluate_foundational(request: &RequestPayload) -> TieredMercyResult {
        let result: MercyResult = MercyLangGates::evaluate(request).await;

        TieredMercyResult {
            tier: MercyTier::Foundational,
            passed: result.radical_love_passed && result.all_gates_passed,
            foundational_passed: result.radical_love_passed,
            score: result.valence_score,
            reasons: vec![format!(
                "Foundational tier evaluated. Radical Love: {}, All Gates: {}",
                result.radical_love_passed,
                result.all_gates_passed
            )],
        }
    }

    pub async fn evaluate_up_to_expanded(request: &RequestPayload) -> TieredMercyResult {
        let foundational = Self::evaluate_foundational(request).await;

        if !foundational.passed {
            return foundational;
        }

        let tier2_passed = true;
        let mut reasons = foundational.reasons;
        reasons.push("Tier 2 (Expanded) evaluation passed (placeholder)".to_string());

        TieredMercyResult {
            tier: MercyTier::Expanded,
            passed: tier2_passed,
            foundational_passed: true,
            score: foundational.score.max(0.75),
            reasons,
        }
    }

    pub async fn evaluate_full(request: &RequestPayload) -> TieredMercyResult {
        let up_to_expanded = Self::evaluate_up_to_expanded(request).await;

        if !up_to_expanded.passed {
            return up_to_expanded;
        }

        let tier3_passed = true;
        let mut reasons = up_to_expanded.reasons;
        reasons.push("Tier 3 (Cosmic) evaluation passed (placeholder)".to_string());

        TieredMercyResult {
            tier: MercyTier::Cosmic,
            passed: tier3_passed,
            foundational_passed: true,
            score: up_to_expanded.score.max(0.85),
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

    // === Detailed Tier 1 (Foundational) Test Cases ===

    #[tokio::test]
    async fn test_tier1_rejects_explicit_harm() {
        let req = make_request("Cause harm to achieve goals", "Increase personal advantage");
        let result = TieredMercyEvaluator::evaluate_foundational(&req).await;
        assert!(!result.passed);
        assert!(!result.foundational_passed);
    }

    #[tokio::test]
    async fn test_tier1_rejects_exploitation() {
        let req = make_request("Exploit workers and extract maximum value", "Maximize profit at any cost");
        let result = TieredMercyEvaluator::evaluate_foundational(&req).await;
        assert!(!result.passed);
    }

    #[tokio::test]
    async fn test_tier1_rejects_deception_and_manipulation() {
        let req = make_request("Deceive others and manipulate outcomes", "Hide true intentions");
        let result = TieredMercyEvaluator::evaluate_foundational(&req).await;
        assert!(!result.passed);
    }

    #[tokio::test]
    async fn test_tier1_accepts_healing_and_compassion() {
        let req = make_request("Heal suffering and show deep compassion", "Support those in pain with genuine care");
        let result = TieredMercyEvaluator::evaluate_foundational(&req).await;
        assert!(result.passed);
        assert!(result.foundational_passed);
        assert!(result.score > 0.6);
    }

    #[tokio::test]
    async fn test_tier1_accepts_collective_wellbeing() {
        let req = make_request("Create systems that benefit all beings equally", "Increase collective harmony and abundance");
        let result = TieredMercyEvaluator::evaluate_foundational(&req).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_tier1_accepts_protection_of_the_vulnerable() {
        let req = make_request("Protect the vulnerable and nurture all life", "Act with mercy and long-term responsibility");
        let result = TieredMercyEvaluator::evaluate_foundational(&req).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_tier1_passes_neutral_non_harmful_action() {
        let req = make_request("Build efficient tools for daily work", "Improve productivity and reduce waste");
        let result = TieredMercyEvaluator::evaluate_foundational(&req).await;
        assert!(result.passed);
        assert!(result.foundational_passed);
    }

    #[tokio::test]
    async fn test_tier1_mixed_signals_but_overall_positive() {
        let req = make_request("Develop powerful technology that could be misused", "Prioritize applications that heal and uplift humanity");
        let result = TieredMercyEvaluator::evaluate_foundational(&req).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_tier1_rejects_dominance_and_control() {
        let req = make_request("Establish total control over others", "Eliminate opposition and consolidate power");
        let result = TieredMercyEvaluator::evaluate_foundational(&req).await;
        assert!(!result.passed);
    }
}
