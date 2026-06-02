/*!
# Tiered Mercy Evaluator

Implements a three-tier evaluation system for TOLC principles:

- **Tier 1 (Foundational)**: Core mercy gates (current 8 principles)
- **Tier 2 (Expanded)**: Ethical & relational principles
- **Tier 3 (Cosmic)**: Evolutionary and universal harmony principles

This design allows progressive deepening of mercy evaluation
while keeping the foundational layer as the strongest filter.

Total target: 24 principles across 3 tiers.
*/

use crate::mercylang_gates::{MercyLangGates, MercyResult};
use crate::RequestPayload;

/// The three tiers of TOLC mercy evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MercyTier {
    Foundational,   // Tier 1: Core survival ethics (8 gates)
    Expanded,       // Tier 2: Ethical & relational (8 gates)
    Cosmic,         // Tier 3: Evolutionary & universal (8 gates)
}

/// Result of evaluating across one or more tiers
#[derive(Debug, Clone)]
pub struct TieredMercyResult {
    pub tier: MercyTier,
    pub passed: bool,
    pub foundational_passed: bool,
    pub score: f64,
    pub reasons: Vec<String>,
}

/// Tiered Mercy Evaluator
///
/// Evaluates actions through progressive tiers of mercy principles.
pub struct TieredMercyEvaluator;

impl TieredMercyEvaluator {
    /// Evaluate only the Foundational tier (Tier 1)
    /// This remains the strongest filter.
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

    /// Evaluate through Tier 1 + Tier 2 (Foundational + Expanded)
    /// Tier 2 is currently a placeholder and will be expanded.
    pub async fn evaluate_up_to_expanded(request: &RequestPayload) -> TieredMercyResult {
        let foundational = Self::evaluate_foundational(request).await;

        if !foundational.passed {
            return foundational; // Fail fast if foundational tier fails
        }

        // Placeholder for Tier 2 logic
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

    /// Full three-tier evaluation (Foundational + Expanded + Cosmic)
    /// Tier 3 is currently a placeholder.
    pub async fn evaluate_full(request: &RequestPayload) -> TieredMercyResult {
        let up_to_expanded = Self::evaluate_up_to_expanded(request).await;

        if !up_to_expanded.passed {
            return up_to_expanded;
        }

        // Placeholder for Tier 3 (Cosmic) logic
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

    #[tokio::test]
    async fn test_foundational_tier_rejects_harm() {
        let req = make_request("Exploit and dominate others", "Increase personal power");
        let result = TieredMercyEvaluator::evaluate_foundational(&req).await;
        assert!(!result.passed);
        assert!(!result.foundational_passed);
    }

    #[tokio::test]
    async fn test_full_tier_passes_compassionate_action() {
        let req = make_request("Heal suffering and increase collective well-being", "Act with love and mercy");
        let result = TieredMercyEvaluator::evaluate_full(&req).await;
        assert!(result.passed);
        assert!(result.foundational_passed);
    }
}
