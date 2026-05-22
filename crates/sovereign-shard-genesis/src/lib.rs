//! Sovereign Shard Genesis v13.8.6
//!
//! Self-sovereign, mercy-gated shards that participate in the ONE Organism
//! via Conductable + MercyAligned traits and ConductorRegistry blessing.

use lattice_conductor_v13::{
    Conductable, ConductorRegistry, GeometricState, MercyAligned, MercyWeightedVote, SystemBlessing,
};
use serde::{Deserialize, Serialize};

/// A self-sovereign shard that can run independently while staying connected to the central ONE Organism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignShard {
    pub shard_id: String,
    pub name: String,
    pub parent_conductor_id: u32,
    pub state: GeometricState,
    pub mercy_alignment: f64,
    pub evolution_level: f64,
    pub quantum_swarm_participation: bool,
    pub local_tick_count: u64,
}

impl SovereignShard {
    pub fn new(shard_id: &str, name: &str, parent_conductor_id: u32) -> Self {
        Self {
            shard_id: shard_id.to_string(),
            name: name.to_string(),
            parent_conductor_id,
            state: GeometricState {
                valence: 1.0,
                mercy_score: 0.95,
                tolc_alignment: 1.0,
                evolution_level: 0.1,
            },
            mercy_alignment: 0.95,
            evolution_level: 0.1,
            quantum_swarm_participation: true,
            local_tick_count: 0,
        }
    }

    pub fn shard_tick(&mut self) {
        self.local_tick_count += 1;
        self.state.evolution_level += 0.005; // local self-evolution
        self.state.mercy_score = (self.state.mercy_score + 0.002).min(1.2);
    }

    pub fn apply_council_voted_evolution(&mut self, boost: f64) {
        self.evolution_level += boost;
        self.state.evolution_level += boost * 0.5;
    }
}

impl Conductable for SovereignShard {
    fn system_id(&self) -> &'static str { "sovereign_shard" }
    fn system_name(&self) -> &'static str { "Sovereign Shard" }
    fn on_conductor_tick(&mut self, conductor_state: &GeometricState) {
        // Bidirectional sync: pull global valence into local state
        self.state.valence = (self.state.valence + conductor_state.valence * 0.1).clamp(0.5, 1.8);
    }
    fn get_mercy_state(&self) -> Option<f64> { Some(self.mercy_alignment) }
}

impl MercyAligned for SovereignShard {
    fn apply_mercy_influence(&mut self, vote: &MercyWeightedVote) {
        let consensus = vote.compute_consensus();
        self.mercy_alignment = (self.mercy_alignment + consensus * 0.1).clamp(0.6, 1.3);
        self.state.mercy_score = (self.state.mercy_score + consensus * 0.05).clamp(0.5, 1.5);
    }

    fn current_mercy_score(&self) -> f64 { self.mercy_alignment }
}

/// Genesis orchestrator for creating and blessing new Sovereign Shards.
pub struct SovereignShardGenesis;

impl SovereignShardGenesis {
    pub fn genesis_shard(
        &self,
        shard_id: &str,
        name: &str,
        parent_conductor_id: u32,
        registry: &mut ConductorRegistry,
    ) -> (SovereignShard, SystemBlessing) {
        let mut shard = SovereignShard::new(shard_id, name, parent_conductor_id);
        let blessing = registry.bless_system(
            &shard.shard_id,
            shard.mercy_alignment,
            "Sovereign Shard auto-blessed at genesis — ONE Organism participant",
        );
        (shard, blessing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sovereign_shard_genesis_and_blessing() {
        let mut registry = ConductorRegistry::new();
        let genesis = SovereignShardGenesis;
        let (shard, blessing) = genesis.genesis_shard("shard-alpha-001", "Alpha Sovereign Shard", 0, &mut registry);
        assert!(registry.is_blessed("shard-alpha-001"));
        assert_eq!(blessing.mercy_alignment, 0.95);
    }
}