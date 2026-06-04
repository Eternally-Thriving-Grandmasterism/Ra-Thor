//! ShardManager — Mercy-Gated Interest Management & Council Proposal Routing
//!
//! Integrates EpigeneticModulation, CouncilProposal evaluation, and real PATSAGi Council valence
//! into shard/interest management for Powrush simulation and ONE Organism coordination.
//!
//! This fulfills the ShardManager integration request by wiring the council simulation
//! directly into spatial interest routing and proposal decisions.

use crate::types::{CouncilProposal, EpigeneticBlessing, EpigeneticModulation};
use crate::riemannian_mercy_manifold::RiemannianMercyManifold;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct InterestSet {
    pub shard_id: String,
    pub entities: Vec<String>,
    pub epigenetic_mod: EpigeneticModulation,
    pub council_scope: String,
}

impl InterestSet {
    pub fn new(shard_id: &str, council_scope: &str) -> Self {
        Self {
            shard_id: shard_id.to_string(),
            entities: Vec::new(),
            epigenetic_mod: EpigeneticModulation::new(0.75, 0.5, "Hyperbolic"),
            council_scope: council_scope.to_string(),
        }
    }

    pub fn add_entity(&mut self, entity_id: &str) {
        self.entities.push(entity_id.to_string());
    }
}

#[derive(Debug, Clone)]
pub struct ShardManager {
    pub shards: HashMap<String, InterestSet>,
    pub manifold: RiemannianMercyManifold,
}

impl ShardManager {
    pub fn new() -> Self {
        Self {
            shards: HashMap::new(),
            manifold: RiemannianMercyManifold::new(),
        }
    }

    pub fn create_shard(&mut self, shard_id: &str, council_scope: &str) {
        let interest_set = InterestSet::new(shard_id, council_scope);
        self.shards.insert(shard_id.to_string(), interest_set);
    }

    /// Core integration: Route a CouncilProposal through real council evaluation + epigenetic modulation.
    /// Returns whether the proposal is accepted into the shard and any epigenetic blessings applied.
    pub fn route_council_proposal(&mut self, proposal: CouncilProposal) -> (bool, Vec<EpigeneticBlessing>, String) {
        // Use the manifold's wired evaluation (which applies to epigenetic_state)
        let (modulated_mercy, blessings, reason) = self.manifold.evaluate_council_proposal(&proposal);

        // Find or create target shard
        let target_shard = if self.shards.contains_key(&proposal.geometric_layer) {
            proposal.geometric_layer.clone()
        } else {
            "default".to_string()
        };

        if let Some(interest_set) = self.shards.get_mut(&target_shard) {
            // Apply the same valence to the shard's local epigenetic state
            interest_set.epigenetic_mod.apply_council_valence(modulated_mercy, &proposal.council);
            interest_set.add_entity(&proposal.proposal_id);
        }

        let accepted = modulated_mercy > 0.85; // Simple threshold; can be made more sophisticated

        (accepted, blessings, reason)
    }

    /// Apply a full council sequence to a specific shard's epigenetic state (for simulation ticks)
    pub fn apply_sequence_to_shard(&mut self, shard_id: &str, sequence: &[(f64, &str)]) -> Option<String> {
        if let Some(interest_set) = self.shards.get_mut(shard_id) {
            let report = interest_set.epigenetic_mod.simulate_council_sequence(sequence);
            Some(report)
        } else {
            None
        }
    }

    pub fn get_shard_summary(&self, shard_id: &str) -> Option<String> {
        self.shards.get(shard_id).map(|set| {
            format!(
                "Shard {} | Entities: {} | Strength: {:.3} | Volatility: {:.3} | Evolution Bonus: {:.3}",
                set.shard_id,
                set.entities.len(),
                set.epigenetic_mod.strength,
                set.epigenetic_mod.volatility,
                set.epigenetic_mod.evolution_rate_bonus()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_manager_routing() {
        let mut manager = ShardManager::new();
        manager.create_shard("hyperbolic_core", "evolutionary");

        let proposal = CouncilProposal::new("prop_001", "evolutionary", "Expand interest in hyperbolic region", "Hyperbolic");
        let (accepted, _blessings, reason) = manager.route_council_proposal(proposal);

        assert!(accepted);
        assert!(reason.contains("evolutionary"));
    }

    #[test]
    fn test_sequence_on_shard() {
        let mut manager = ShardManager::new();
        manager.create_shard("test_shard", "harmony");

        let seq = vec![(0.92, "harmony"), (0.95, "truth")];
        let report = manager.apply_sequence_to_shard("test_shard", &seq).unwrap();

        assert!(report.contains("Cumulative"));
    }
}
