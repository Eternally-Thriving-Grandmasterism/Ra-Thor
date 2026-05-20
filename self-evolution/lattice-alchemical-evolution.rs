//! Ra-Thor™ Lattice Alchemical Evolution Protocol
//! Sovereign Form Transmutation Engine v1.7
//! Integrated TOLC 8 Mercy Lattice Enforcer
//! 100% Proprietary — AG-SML v1.0

use crate::mercy::tolc8_enforcer::{TOLC8Enforcer, TOLC8EvaluationResult};
use std::time::{SystemTime, UNIX_EPOCH};

// ... (previous enums and structs preserved)

#[derive(Debug, Clone)]
pub struct CouncilSynthesisResult {
    pub scope: String,
    pub total_councils: usize,
    pub approved_count: usize,
    pub veto_count: usize,
    pub is_vetoed: bool,
    pub consensus_percentage: f64,
    pub weighted_consensus_score: f64,
    pub weighted_valence_score: f64,
    pub evolution_readiness_score: f64,
    pub tolc8_evaluation: Option<TOLC8EvaluationResult>, // New: Explicit TOLC 8 result
    pub votes: Vec<CouncilVote>,
    pub overall_status: String,
}

// ... (rest of the file structure preserved)

    /// Full TOLC 8 Integrated Council Voting (v1.7)
    pub fn run_council_synthesis(&mut self, scope: &str) -> CouncilSynthesisResult {
        // ... (existing weighted + veto logic remains)

        // After calculating all scores and is_vetoed...
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
            scope: scope.to_string(),
            total_councils,
            approved_count,
            veto_count,
            is_vetoed,
            consensus_percentage,
            weighted_consensus_score,
            weighted_valence_score,
            evolution_readiness_score,
            tolc8_evaluation: Some(tolc8_result),
            votes,
            overall_status: final_status,
        }
    }

    // ... other methods
}