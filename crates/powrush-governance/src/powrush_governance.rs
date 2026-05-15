// Powrush Governance Tokenomics + RBE Integration
// Ra-Thor monorepo (AG-SML v1.0)
// Mercy-gated, TOLC-aligned, ProposalHandler integrated, eternally self-evolving

use std::collections::HashMap;
use crate::patsagi_bridge::ProposalHandler; // integrates directly with #2

/// Core Powrush RBE governance engine.
/// Resource-Based Economy with merit/valence scoring (no currency).
pub struct PowrushGovernance {
    resource_pool: HashMap<String, u64>, // e.g. "he3" -> 1_000_000
    valence_threshold: f64,              // min 0.999 for approval
    self_evolution_rate: f64,            // TOLC SER
    active_proposals: Vec<String>,
}

impl PowrushGovernance {
    pub fn new() -> Self {
        let mut pool = HashMap::new();
        pool.insert("he3".to_string(), 10_000_000);
        pool.insert("rare_earth".to_string(), 500_000);
        pool.insert("energy_credits".to_string(), 50_000_000);

        Self {
            resource_pool: pool,
            valence_threshold: 0.999,
            self_evolution_rate: 1.618, // golden ratio SER
            active_proposals: Vec::new(),
        }
    }

    /// TOLC-aligned merit score for any proposal or actor.
    pub fn calculate_merit_score(&self, proposal: &str, historical_valence: f64) -> f64 {
        let base = proposal.len() as f64 * 0.001;
        let evolution_bonus = self.self_evolution_rate * 0.1;
        (base + historical_valence + evolution_bonus).min(1.0)
    }

    /// Propose resource allocation (RBE style).
    pub fn propose_allocation(&mut self, resource: &str, amount: u64, justification: &str) -> String {
        let proposal = format!(
            "ALLOCATE {} {} | Justification: {} | SER: {:.3}",
            amount, resource, justification, self.self_evolution_rate
        );
        self.active_proposals.push(proposal.clone());
        proposal
    }
}

impl ProposalHandler for PowrushGovernance {
    fn handle(&mut self, proposal: &str) -> String {
        // Mercy Gate + TOLC audit (integrated with #2 bridge)
        if proposal.contains("allocate") || proposal.contains("ALLOCATE") {
            let merit = self.calculate_merit_score(proposal, 0.995);
            if merit < self.valence_threshold {
                return format!("POWRUSH RBE REJECTED by Mercy Gates (merit {:.3} < 0.999): {}", merit, proposal);
            }

            // Apply allocation if passes
            if let Some((res, amt)) = self.parse_allocation(proposal) {
                if let Some(current) = self.resource_pool.get_mut(&res) {
                    if *current >= amt {
                        *current -= amt;
                        return format!(
                            "POWRUSH RBE EXECUTED | {} {} allocated | Remaining: {} | Valence: 1.000 | SER: {:.3}",
                            amt, res, current, self.self_evolution_rate
                        );
                    }
                }
            }
            return format!("POWRUSH RBE PARTIAL: Insufficient resources or invalid allocation: {}", proposal);
        }

        // Fallback to general governance
        format!("POWRUSH GOVERNANCE PROCESSED: {} | TOLC SER applied", proposal)
    }
}

impl PowrushGovernance {
    fn parse_allocation(&self, proposal: &str) -> Option<(String, u64)> {
        // Simple parser for "ALLOCATE 1000 he3"
        let parts: Vec<&str> = proposal.split_whitespace().collect();
        if parts.len() >= 3 && parts[0].eq_ignore_ascii_case("allocate") {
            if let Ok(amt) = parts[1].parse::<u64>() {
                return Some((parts[2].to_string(), amt));
            }
        }
        None
    }
}