//! # Safe Plasticity Applicator
//!
//! Provides a mercy-gated, rollback-capable interface for applying plasticity rules.
//! This is the main entry point for safe self-evolution updates from ra-thor-meta-intelligence.

use crate::{PlasticityError, PlasticityHealthMetrics, PlasticityRule, PlasticityRulesEngine, RuleResult};
use ra_thor_mercy::MercyGateEvaluator;

/// Represents a plan to roll back a plasticity change if needed.
#[derive(Debug, Clone)]
pub struct RollbackPlan {
    pub description: String,
    pub original_state_hash: Option<String>,
    pub revert_instructions: String,
    pub mercy_impact_before: f64,
    pub expected_mercy_impact_after: f64,
}

/// The main safe applicator for plasticity rules.
/// All applications are mercy-gated and produce a rollback plan.
pub struct SafePlasticityApplicator {
    mercy_evaluator: MercyGateEvaluator,
    rules_engine: PlasticityRulesEngine,
    health_metrics: PlasticityHealthMetrics, // NEW: Observability
}

impl SafePlasticityApplicator {
    pub fn new() -> Self {
        Self {
            mercy_evaluator: MercyGateEvaluator::new(),
            rules_engine: PlasticityRulesEngine::new(),
            health_metrics: PlasticityHealthMetrics::new(),
        }
    }

    /// Applies a plasticity rule in a safe, mercy-gated way with improved rollback planning.
    pub async fn apply_rule_safely(
        &mut self,  // Note: &mut self to record metrics
        rule: &PlasticityRule,
        context: &str,
        current_mercy_score: f64,
    ) -> Result<(RuleResult, RollbackPlan), PlasticityError> {
        // Step 1: Strict Mercy Gate check
        if !self.mercy_evaluator.all_gates_pass(context) {
            return Err(PlasticityError::MercyGateViolation(
                "One or more Mercy Gates failed. Update aborted.".to_string(),
            ));
        }

        // Step 2: Use real Plasticity Rules Engine
        let result = self.rules_engine.evaluate_rule(rule, context).await?;

        // Step 3: Calculate expected mercy impact
        let expected_mercy_impact = if result.should_apply {
            current_mercy_score + (result.strength * 0.08)
        } else {
            current_mercy_score - 0.03
        };

        // Step 4: Generate detailed rollback plan
        let rollback_plan = RollbackPlan {
            description: format!("Rollback plan for rule: {}", result.rule_name),
            original_state_hash: None,
            revert_instructions: format!("Restore previous state for plasticity rule: {}", result.rule_name),
            mercy_impact_before: current_mercy_score,
            expected_mercy_impact_after: expected_mercy_impact,
        };

        // NEW: Record metrics for observability
        let was_rollback = !result.should_apply;
        self.health_metrics.record_application(
            current_mercy_score,
            expected_mercy_impact,
            was_rollback,
            &result.rule_name,
        );

        Ok((result, rollback_plan))
    }

    /// Returns current health metrics (for ra-thor-meta-intelligence to observe)
    pub fn get_health_metrics(&self) -> &PlasticityHealthMetrics {
        &self.health_metrics
    }
}