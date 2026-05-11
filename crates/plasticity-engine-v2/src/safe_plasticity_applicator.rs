//! # Safe Plasticity Applicator
//!
//! Provides a mercy-gated, rollback-capable interface for applying plasticity rules.
//! This is the main entry point for safe self-evolution updates from ra-thor-meta-intelligence.

use crate::{PlasticityError, PlasticityRule, PlasticityRulesEngine, RuleResult};
use ra_thor_mercy::MercyGateEvaluator;

/// Represents a plan to roll back a plasticity change if needed.
#[derive(Debug, Clone)]
pub struct RollbackPlan {
    pub description: String,
    pub original_state_hash: Option<String>,
    pub revert_instructions: String,
}

/// The main safe applicator for plasticity rules.
/// All applications are mercy-gated and produce a rollback plan.
pub struct SafePlasticityApplicator {
    mercy_evaluator: MercyGateEvaluator,
    rules_engine: PlasticityRulesEngine,
}

impl SafePlasticityApplicator {
    pub fn new() -> Self {
        Self {
            mercy_evaluator: MercyGateEvaluator::new(),
            rules_engine: PlasticityRulesEngine::new(),
        }
    }

    /// Applies a plasticity rule in a safe, mercy-gated way.
    /// Now uses the real PlasticityRulesEngine for decision making.
    pub async fn apply_rule_safely(
        &self,
        rule: &PlasticityRule,
        context: &str,
        impact: Option<&ra_thor_legal_lattice::cehi::CEHIImpact>,
    ) -> Result<(RuleResult, RollbackPlan), PlasticityError> {
        // Step 1: Check all 7 Living Mercy Gates
        if !self.mercy_evaluator.all_gates_pass(context) {
            return Err(PlasticityError::MercyGateViolation(
                "One or more Mercy Gates failed. Update aborted.".to_string(),
            ));
        }

        // Step 2: Use real Plasticity Rules Engine if impact data is available
        let result = if let Some(impact_data) = impact {
            self.rules_engine.evaluate(impact_data).await?
        } else {
            // Fallback for cases without full CEHI data
            RuleResult {
                rule_name: format!("{:?}", rule),
                should_apply: true,
                strength: 0.75,
            }
        };

        // Step 3: Generate rollback plan
        let rollback_plan = RollbackPlan {
            description: format!("Rollback plan for rule: {}", result.rule_name),
            original_state_hash: None, // TODO: Compute real state hash
            revert_instructions: format!("Restore previous state for rule: {}", result.rule_name),
        };

        Ok((result, rollback_plan))
    }
}