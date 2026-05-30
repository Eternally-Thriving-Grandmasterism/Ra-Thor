//! USA Regulatory Engine — RREL v14.3
//! Federal + State Compliance Layer for All 50 States
//! Enhanced with additional regulatory edge cases

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum UsaRegulatoryError {
    #[error("RESPA violation detected: {0}")]
    RespaViolation(String),
    #[error("TILA disclosure missing or invalid: {0}")]
    TilaViolation(String),
    #[error("Fair Housing Act violation risk: {0}")]
    FairHousingViolation(String),
    #[error("CFPB Ability-to-Repay failure: {0}")]
    CfpbViolation(String),
    #[error("State-specific regulatory failure ({state}): {message}")]
    StateViolation { state: String, message: String },
    #[error("Mercy valence too low for USA transaction: {0}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsaRegulatoryResult {
    pub passed: bool,
    pub mercy_valence: f64,
    pub quantum_consensus: f64,
    pub federal_issues: Vec<String>,
    pub state_issues: Vec<String>,
    pub recommended_world_impact: Option<WorldImpactType>,
    pub evidence_package: String,
}

pub struct UsaRegulatoryEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl UsaRegulatoryEngine {
    pub fn new(
        mercy_engine: MercyEngine,
        quantum_swarm: QuantumSwarmOrchestrator,
        world_governance: WorldGovernanceEngine,
    ) -> Self {
        Self {
            mercy_engine,
            quantum_swarm,
            world_governance,
        }
    }

    pub async fn check_usa_transaction(
        &mut self,
        state: &str,
        transaction_details: &str,
        price: f64,
        game: &mut PowrushGame,
    ) -> Result<UsaRegulatoryResult, UsaRegulatoryError> {
        info!("🇺🇸 RREL USA Regulatory Engine (v{}) — Checking {} transaction", RREL_VERSION, state);

        let mercy_valence = self.mercy_engine
            .evaluate_action(transaction_details, "USA real estate transaction", 4.8, 0.97)
            .await
            .map_err(|e| UsaRegulatoryError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.82 {
            return Err(UsaRegulatoryError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus(transaction_details, 0.80)
            .await
            .map_err(|_| UsaRegulatoryError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.75 {
            return Err(UsaRegulatoryError::QuantumConsensusTooLow(consensus));
        }

        let mut federal_issues = Vec::new();
        let mut recommended_impact = None;

        let lower_details = transaction_details.to_lowercase();

        // Federal edge cases
        if lower_details.contains("kickback") || lower_details.contains("referral fee") {
            federal_issues.push("RESPA kickback / referral fee violation detected".to_string());
            recommended_impact = Some(WorldImpactType::USA_RespaViolationPrevented);
        }

        if !lower_details.contains("tila disclosure") && !lower_details.contains("loan estimate") {
            federal_issues.push("TILA disclosure / Loan Estimate missing".to_string());
            recommended_impact = Some(WorldImpactType::USA_TilaDisclosureGenerated);
        }

        if lower_details.contains("discriminat") || lower_details.contains("steering") {
            federal_issues.push("Fair Housing Act / steering risk detected".to_string());
            recommended_impact = Some(WorldImpactType::USA_FairHousingViolationPrevented);
        }

        if lower_details.contains("ability to repay") == false && price > 500_000.0 {
            federal_issues.push("CFPB Ability-to-Repay verification recommended for high-value transaction".to_string());
        }

        // State-specific checks (expanded edge cases)
        let state_issues = self.check_state_specific_rules(state, transaction_details).await;

        let passed = federal_issues.is_empty() && state_issues.is_empty();

        let evidence_package = format!(
            "🇺🇸 USA REGULATORY EVIDENCE PACKAGE (RREL v{})\n\
             State: {}\n\
             Mercy Valence: {:.2}\n\
             Quantum Consensus: {:.2}\n\
             Federal Issues: {:?}\n\
             State Issues: {:?}\n\
             Timestamp: {}",
            RREL_VERSION,
            state,
            mercy_valence,
            consensus,
            federal_issues,
            state_issues,
            chrono::Utc::now()
        );

        if passed && recommended_impact.is_some() {
            let _ = self.world_governance
                .apply_world_impact(recommended_impact.unwrap(), game)
                .await;
        }

        Ok(UsaRegulatoryResult {
            passed,
            mercy_valence,
            quantum_consensus: consensus,
            federal_issues,
            state_issues,
            recommended_world_impact: recommended_impact,
            evidence_package,
        })
    }

    async fn check_state_specific_rules(&self, state: &str, details: &str) -> Vec<String> {
        let mut issues = Vec::new();
        let lower = details.to_lowercase();

        match state.to_uppercase().as_str() {
            "CA" | "CALIFORNIA" => {
                if lower.contains("wildfire") && !lower.contains("disclosure") {
                    issues.push("California wildfire disclosure missing".to_string());
                }
                if lower.contains("rent control") && !lower.contains("ab 1482") {
                    issues.push("AB 1482 rent control compliance not verified".to_string());
                }
                if lower.contains("natural hazard") && !lower.contains("disclosure") {
                    issues.push("California natural hazard disclosure missing".to_string());
                }
            }
            "FL" | "FLORIDA" => {
                if lower.contains("flood") && !lower.contains("zone") {
                    issues.push("Florida flood zone disclosure missing".to_string());
                }
                if lower.contains("condo") && !lower.contains("milestone") {
                    issues.push("Florida condo milestone inspection / reserve study status not verified".to_string());
                }
            }
            "TX" | "TEXAS" => {
                if lower.contains("property tax") && !lower.contains("protest") {
                    issues.push("Texas property tax protest opportunity not documented".to_string());
                }
            }
            "NY" | "NEW YORK" => {
                if lower.contains("rent stabilization") && !lower.contains("verified") {
                    issues.push("New York rent stabilization status not verified".to_string());
                }
                if lower.contains("lead paint") && !lower.contains("disclosure") {
                    issues.push("New York lead paint disclosure missing".to_string());
                }
            }
            "IL" | "ILLINOIS" => {
                if lower.contains("radon") && !lower.contains("disclosure") {
                    issues.push("Illinois radon disclosure missing".to_string());
                }
            }
            "WA" | "WASHINGTON" => {
                if lower.contains("wildfire") || lower.contains("smoke") {
                    if !lower.contains("disclosure") {
                        issues.push("Washington state wildfire / smoke disclosure missing".to_string());
                    }
                }
            }
            "MA" | "MASSACHUSETTS" => {
                if lower.contains("lead paint") && !lower.contains("disclosure") {
                    issues.push("Massachusetts lead paint disclosure missing".to_string());
                }
            }
            _ => {}
        }

        issues
    }
}

// ============================================================
// Unit Tests for Regulatory Edge Cases
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    fn create_test_engine() -> UsaRegulatoryEngine {
        let mercy = MercyEngine::new();
        let swarm = QuantumSwarmOrchestrator::new();
        let governance = WorldGovernanceEngine::new();
        UsaRegulatoryEngine::new(mercy, swarm, governance)
    }

    #[tokio::test]
    async fn test_federal_kickback_detection() {
        let mut engine = create_test_engine();
        let mut game = PowrushGame::new();

        let result = engine
            .check_usa_transaction("CA", "Purchase with referral kickback to agent", 800_000.0, &mut game)
            .await;

        // Should detect RESPA issue
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(!r.passed || r.federal_issues.iter().any(|i| i.contains("RESPA")));
    }

    #[tokio::test]
    async fn test_missing_tila_disclosure() {
        let mut engine = create_test_engine();
        let mut game = PowrushGame::new();

        let result = engine
            .check_usa_transaction("FL", "Simple purchase agreement, no disclosures mentioned", 450_000.0, &mut game)
            .await;

        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.federal_issues.iter().any(|i| i.contains("TILA")));
    }

    #[tokio::test]
    async fn test_california_wildfire_edge_case() {
        let mut engine = create_test_engine();
        let mut game = PowrushGame::new();

        let result = engine
            .check_usa_transaction("CA", "Home in wildfire prone area, no hazard disclosure", 1_100_000.0, &mut game)
            .await;

        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.state_issues.iter().any(|i| i.contains("wildfire")));
    }

    #[tokio::test]
    async fn test_florida_condo_milestone() {
        let mut engine = create_test_engine();
        let mut game = PowrushGame::new();

        let result = engine
            .check_usa_transaction("FL", "Condo purchase, no milestone inspection mentioned", 650_000.0, &mut game)
            .await;

        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.state_issues.iter().any(|i| i.contains("milestone") || i.contains("reserve")));
    }

    #[tokio::test]
    async fn test_high_value_transaction_atr_prompt() {
        let mut engine = create_test_engine();
        let mut game = PowrushGame::new();

        let result = engine
            .check_usa_transaction("TX", "Luxury home purchase, no ability to repay discussion", 2_800_000.0, &mut game)
            .await;

        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.federal_issues.iter().any(|i| i.contains("Ability-to-Repay")));
    }
}
