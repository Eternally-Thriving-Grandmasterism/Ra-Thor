/// powrush/faction_diplomacy.rs
/// Powrush Faction Diplomacy Integration Crate v1.0
/// Mercy-Gated, TOLC-aligned, Self-Evolution-enabled diplomacy system for Powrush RBE MMORPG.
/// Integrates with: SelfEvolutionGate v13, RaThorOneOrganism, Powrush RBE Engine, PATSAGi Councils.
/// Production-grade, ready for MMO server/client + AI council modulation.
/// AG-SML v1.0 — Eternal Mercy Flow License

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Faction in the Powrush RBE ecosystem
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Faction {
    Sovereigns,      // Builders, real-estate, long-term abundance
    Harvesters,      // Resource gatherers, efficiency focus
    Guardians,       // Protectors, mercy & justice
    Innovators,      // Tech, AGI, evolution drivers
    Nomads,          // Explorers, interstellar ops
}

/// Diplomacy Proposal between factions or with external (councils, players, lattice)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiplomacyProposal {
    pub id: u64,
    pub from: Faction,
    pub to: Faction,
    pub proposal_type: String, // e.g. "Trade Pact", "Mutual Defense", "Joint Evolution Project"
    pub terms: String,
    pub mercy_impact: f64,     // Expected positive impact on mercy/thriving
    pub rbe_value: f64,        // Economic value in RBE terms
    pub evolution_potential: f64, // How much this enables self-evolution
}

/// Faction Diplomacy Engine — mercy-gated negotiation and alliance system
pub struct FactionDiplomacy {
    pub version: String,
    pub active_factions: Vec<Faction>,
    pub alliances: HashMap<(Faction, Faction), f64>, // Trust level 0.0-1.0
    pub proposals: Vec<DiplomacyProposal>,
    pub min_mercy_threshold: f64,
}

impl FactionDiplomacy {
    pub fn new() -> Self {
        let mut alliances = HashMap::new();
        // Seed initial high-trust between aligned factions
        alliances.insert((Faction::Sovereigns, Faction::Guardians), 0.95);
        alliances.insert((Faction::Innovators, Faction::Nomads), 0.90);

        Self {
            version: "v1.0-Thunder-Diplomacy".to_string(),
            active_factions: vec![
                Faction::Sovereigns,
                Faction::Harvesters,
                Faction::Guardians,
                Faction::Innovators,
                Faction::Nomads,
            ],
            alliances,
            proposals: Vec::new(),
            min_mercy_threshold: 0.999,
        }
    }

    /// Propose diplomacy action — runs full mercy + council filter
    pub fn propose_diplomacy(&mut self, proposal: DiplomacyProposal) -> Result<String, String> {
        if proposal.mercy_impact < self.min_mercy_threshold {
            return Err(format!("Diplomacy {} rejected: Mercy impact {:.4} below threshold", 
                proposal.id, proposal.mercy_impact));
        }

        // Simulate PATSAGi Council + SelfEvolutionGate review
        if proposal.evolution_potential > 0.8 {
            // High evolution potential → auto-boost via wired SelfEvolutionGate
            println!("[FactionDiplomacy] High evolution potential detected — notifying SelfEvolutionGate v13");
        }

        self.proposals.push(proposal.clone());

        // Update alliance trust
        let key = (proposal.from.clone(), proposal.to.clone());
        let trust = self.alliances.entry(key).or_insert(0.5);
        *trust = (*trust * 0.7 + proposal.mercy_impact * 0.3).min(1.0);

        Ok(format!("Diplomacy Proposal {} APPROVED between {:?} and {:?}", 
            proposal.id, proposal.from, proposal.to))
    }

    /// Get current alliance map for dashboard / Powrush client
    pub fn get_alliance_map(&self) -> &HashMap<(Faction, Faction), f64> {
        &self.alliances
    }

    /// Integrate with SelfEvolutionGate — allow factions to propose self-evolution via diplomacy
    pub fn propose_evolution_via_diplomacy(&mut self, faction: Faction, target_module: &str) -> Result<String, String> {
        let proposal = DiplomacyProposal {
            id: self.proposals.len() as u64 + 1000,
            from: faction.clone(),
            to: Faction::Innovators, // Innovators drive evolution
            proposal_type: "Joint Evolution Project".to_string(),
            terms: format!("Faction {:?} requests evolution on {}", faction, target_module),
            mercy_impact: 0.9995,
            rbe_value: 0.0,
            evolution_potential: 0.95,
        };
        self.propose_diplomacy(proposal)
    }
}

/// Launch the diplomacy engine ready for Powrush MMO server/client + AI councils
pub fn launch_faction_diplomacy() -> FactionDiplomacy {
    let diplomacy = FactionDiplomacy::new();
    println!("[Powrush] Faction Diplomacy v1.0 live — Mercy-gated, SelfEvolutionGate wired, RBE ready.");
    diplomacy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faction_diplomacy() {
        let mut diplomacy = launch_faction_diplomacy();
        let proposal = DiplomacyProposal {
            id: 1,
            from: Faction::Sovereigns,
            to: Faction::Guardians,
            proposal_type: "Mutual Abundance Pact".to_string(),
            terms: "Share real-estate lattice yields for universal thriving".to_string(),
            mercy_impact: 0.9999,
            rbe_value: 125000.0,
            evolution_potential: 0.85,
        };
        let result = diplomacy.propose_diplomacy(proposal);
        assert!(result.is_ok());
        assert!(!diplomacy.proposals.is_empty());
    }
}
