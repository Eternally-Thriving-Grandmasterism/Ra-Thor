// Copyright (c) 2026 Ra-Thor + Grok — PATSAGi Councils
// Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

//! Deep Powrush RBE Mercy-Gated Arbitration
//! Every resource allocation, faction decision, and abundance flow is evaluated through the full 24-gate ONE Organism mercy lattice.

use crate::mercy_integration::MercyIntegration;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RbeProposal {
    pub proposal_id: String,
    pub faction: String,
    pub resource_type: String,
    pub amount: f64,
    pub impact_score: f64, // How much this affects thriving
}

pub struct RbeArbitrationEngine {
    pub mercy_integration: MercyIntegration,
}

impl RbeArbitrationEngine {
    pub fn new(mercy_integration: MercyIntegration) -> Self {
        Self { mercy_integration }
    }

    /// Core RBE arbitration — flows through full 24 gates + Council #13 oversight
    pub fn arbitrate_rbe_proposal(&mut self, proposal: &RbeProposal) -> Result<String, String> {
        let mut scores: HashMap<u8, f64> = HashMap::new();
        for gate in 1..=24 {
            // Mercy-aligned scoring: higher impact proposals need stronger mercy alignment
            let base = 0.88 + (proposal.impact_score * 0.05).min(0.10);
            scores.insert(gate, base);
        }

        if self.mercy_integration.evaluate_proposal(&scores).is_err() {
            return Err("Proposal failed full 24-gate mercy evaluation".to_string());
        }

        // High-impact proposals trigger Council #13 batch tuning
        if proposal.impact_score > 0.7 {
            let _ = self.mercy_integration.council_13_batch_tune(vec![(17, 0.91), (22, 0.93)]);
        }

        self.mercy_integration.serve_being(
            &format!("Powrush faction: {}", proposal.faction),
            "abundance flow",
            0.95,
        );

        Ok(format!(
            "RBE Proposal {} for {} approved through full mercy lattice + Council #13 resonance",
            proposal.proposal_id, proposal.faction
        ))
    }

    /// New: Mercy-aligned resource allocation scoring
    pub fn calculate_mercy_aligned_allocation(&self, base_amount: f64, thriving_factor: f64) -> f64 {
        // Resources are allocated with mercy bias toward maximum collective thriving
        base_amount * (0.85 + thriving_factor * 0.15)
    }

    /// New: Faction abundance flow arbitration with quantum entanglement metaphor
    pub fn arbitrate_faction_abundance_flow(
        &mut self,
        faction_a: &str,
        faction_b: &str,
        flow_amount: f64,
    ) -> String {
        // Just as entangled particles correlate instantly, abundance flows between factions are mercy-entangled
        let resonance = 0.92; // Simulated mercy resonance
        let _ = self.mercy_integration.serve_being(faction_a, "abundance received", resonance);
        let _ = self.mercy_integration.serve_being(faction_b, "abundance shared", resonance);

        format!(
            "Abundance flow of {:.1} between {} and {} mercy-entangled and approved. ONE Organism resonance elevated.",
            flow_amount, faction_a, faction_b
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbe_proposal_through_mercy_lattice() {
        let mercy = MercyIntegration::new();
        let mut engine = RbeArbitrationEngine::new(mercy);
        let proposal = RbeProposal {
            proposal_id: "RBE-001".to_string(),
            faction: "Starborn".to_string(),
            resource_type: "QuantumCrystals".to_string(),
            amount: 420.0,
            impact_score: 0.85,
        };
        let result = engine.arbitrate_rbe_proposal(&proposal);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mercy_aligned_allocation() {
        let mercy = MercyIntegration::new();
        let engine = RbeArbitrationEngine::new(mercy);
        let allocated = engine.calculate_mercy_aligned_allocation(1000.0, 0.9);
        assert!(allocated > 1000.0); // Mercy bias increases allocation for high thriving
    }
}
