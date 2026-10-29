//! ShardManager — Mercy-Gated Interest Management & Council Proposal Routing
//!
//! Integrates EpigeneticModulation, CouncilProposal evaluation, and real PATSAGi Council valence
//! into shard/interest management for Powrush simulation and ONE Organism coordination.
//!
//! Extended with Particle Evolution wiring for Resonance Gear visual feedback (council-aware / mercy-gated).

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
    pub fn route_council_proposal(&mut self, proposal: CouncilProposal) -> (bool, Vec<EpigeneticBlessing>, String) {
        let (modulated_mercy, blessings, reason) = self.manifold.evaluate_council_proposal(&proposal);

        let target_shard = if self.shards.contains_key(&proposal.geometric_layer) {
            proposal.geometric_layer.clone()
        } else {
            "default".to_string()
        };

        if let Some(interest_set) = self.shards.get_mut(&target_shard) {
            interest_set.epigenetic_mod.apply_council_valence(modulated_mercy, &proposal.council);
            interest_set.add_entity(&proposal.proposal_id);
        }

        let accepted = modulated_mercy > 0.85;
        (accepted, blessings, reason)
    }

    /// NEW: Handle particle evolution events from Resonance Gear (Powrush particles crate).
    /// Creates an internal CouncilProposal for "ParticleEvolution" and routes it.
    /// Returns epigenetic blessings that can be used to modulate burst intensity, color, or lifetime in the Bevy Hanabi layer.
    /// This is the professional wiring point for council-aware / mercy-gated visual feedback.
    pub fn handle_particle_evolution(
        &mut self,
        faction: &str,
        old_level: u32,
        new_level: u32,
        harmony: f64,
    ) -> (bool, Vec<EpigeneticBlessing>, String) {
        let proposal = CouncilProposal::new(
            &format!("particle_evolution_{}_{}", faction, new_level),
            faction,
            &format!("Resonance Gear ({}) evolved from level {} to {} with harmony {:.2}", faction, old_level, new_level, harmony),
            if faction == "Forge" { "Hyperbolic" } else { "Platonic" },
        );

        // Route through full council evaluation + epigenetic modulation
        let result = self.route_council_proposal(proposal);

        // Future: Use result.1 (blessings) to scale burst particle_count, color saturation, or lifetime in bevy_hanabi_plugin
        result
    }

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
    fn test_particle_evolution_wiring() {
        let mut manager = ShardManager::new();
        manager.create_shard("forge_shard", "evolutionary");

        let (accepted, blessings, reason) = manager.handle_particle_evolution("Forge", 2, 3, 0.92);

        assert!(accepted);
        assert!(!blessings.is_empty() || reason.contains("evolved"));
    }
}
