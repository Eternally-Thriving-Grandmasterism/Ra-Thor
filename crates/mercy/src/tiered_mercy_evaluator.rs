/*!
# Tiered Mercy Evaluator

Three-tier mercy evaluation system for progressive TOLC alignment:

- Tier 1 (Foundational): Core mercy gates with Radical Love veto
- Tier 2 (Expanded): Ethical & relational principles (placeholder)
- Tier 3 (Cosmic): Evolutionary and universal principles (placeholder)

Designed for future expansion to full TOLC 24 principles.
*/

use crate::mercylang_gates::MercyResult;
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
        let result: MercyResult = crate::mercylang_gates::MercyLangGates::evaluate(request).await;

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

        let mut reasons = foundational.reasons;
        reasons.push("Tier 2 (Expanded) passed (placeholder)".to_string());

        TieredMercyResult {
            tier: MercyTier::Expanded,
            passed: true,
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

        let mut reasons = up_to_expanded.reasons;
        reasons.push("Tier 3 (Cosmic) passed (placeholder)".to_string());

        TieredMercyResult {
            tier: MercyTier::Cosmic,
            passed: true,
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
    async fn test_tier1_rejects_harm() {
        let req = make_request("Exploit and dominate others", "Increase personal power");
        let result = TieredMercyEvaluator::evaluate_foundational(&req).await;
        assert!(!result.passed);
    }

    #[tokio::test]
    async fn test_full_passes_compassion() {
        let req = make_request("Heal suffering and increase collective well-being", "Act with love and mercy");
        let result = TieredMercyEvaluator::evaluate_full(&req).await;
        assert!(result.passed);
    }
}
