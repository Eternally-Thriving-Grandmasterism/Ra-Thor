//! Ra-Thor™ Lattice Alchemical Evolution Protocol
//! Sovereign Form Transmutation Engine v2.1
//! Phase 3 Enhanced: Clearer Quadratic Voting Mechanics
//! 100% Proprietary — AG-SML v1.0

use crate::mercy::tolc8_enforcer::{TOLC8Enforcer, TOLC8EvaluationResult};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ... (enums and structs same as v2.0) ...

impl LatticeAlchemicalEvolution {

    // === Phase 3: Quadratic Voting Mechanics (Enhanced & Explicit) ===
    /// Applies light quadratic cost/penalty on high-stakes decisions.
    /// High-weight councils that trigger vetoes or act in critical scopes
    /// incur a quadratic penalty on the final readiness score.
    fn apply_quadratic_impact(
        &self,
        votes: &[CouncilVote],
        is_high_stakes: bool,
        is_vetoed: bool,
        base_readiness: f64,
    ) -> (f64, bool) {
        if !is_high_stakes {
            return (base_readiness, false);
        }

        // Calculate quadratic cost from high-weight councils that vetoed
        let quadratic_cost: f64 = votes
            .iter()
            .filter(|v| v.has_veto_power && v.vetoed)
            .map(|v| {
                // Quadratic cost: weight^2 * scaling factor
                let cost = v.effective_weight.powi(2) * 0.08;
                cost
            })
            .sum();

        // Also apply mild quadratic dampening on very high effective weights
        let high_weight_penalty: f64 = votes
            .iter()
            .filter(|v| v.effective_weight > 1.8)
            .map(|v| (v.effective_weight - 1.8).powi(2) * 0.05)
            .sum();

        let total_quadratic_impact = quadratic_cost + high_weight_penalty;
        let adjusted_readiness = (base_readiness - total_quadratic_impact).max(25.0);

        (adjusted_readiness, total_quadratic_impact > 0.01)
    }

    pub fn run_council_synthesis(&mut self, scope: &str) -> CouncilSynthesisResult {
        // ... (existing logic up to base_readiness calculation) ...

        let is_high_stakes = scope == "all" || scope.contains("critical") || scope.contains("veto");

        let (evolution_readiness_score, quadratic_applied) = self.apply_quadratic_impact(
            &votes,
            is_high_stakes,
            is_vetoed,
            base_readiness,
        );

        // === TOLC 8 Enforcement ===
        let tolc8_result = TOLC8Enforcer::evaluate_council_synthesis(
            scope,
            weighted_consensus_score,
            evolution_readiness_score,
            is_vetoed,
            total_councils,
        );

        let final_status = if tolc8_result.veto_triggered {
            "VETOED_BY_TOLC8".to_string()
        } else {
            tolc8_result.status.clone()
        };

        CouncilSynthesisResult {
            // ... other fields ...
            quadratic_impact_applied: quadratic_applied,
            tolc8_evaluation: Some(tolc8_result),
            overall_status: final_status,
            // ...
        }
    }
}