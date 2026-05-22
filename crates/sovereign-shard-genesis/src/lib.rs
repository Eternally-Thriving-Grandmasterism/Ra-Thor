//! Sovereign Shard Genesis v13.8.7
//! Self-sovereign, mercy-aligned shards that can run independently while remaining part of the ONE Organism.

use crate::lattice_conductor_v13::{Conductable, MercyAligned, GeometricState, MercyWeightedVote};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignShard {
    pub shard_id: String,
    pub parent_conductor_id: u32,
    pub state: GeometricState,
    pub mercy_alignment: f64,
    pub evolution_level: f64,
    pub quantum_swarm_participation: f64,
    pub offline_mode: bool,
    pub last_reconciled_tick: u64,
    pub influence_history: Vec<String>,
}

impl SovereignShard {
    pub fn new(shard_id: &str, parent_id: u32, initial_mercy: f64) -> Self {
        Self {
            shard_id: shard_id.to_string(),
            parent_conductor_id: parent_id,
            state: GeometricState { valence: 1.0, mercy_score: initial_mercy, tolc_alignment: 1.0, evolution_level: 0.0 },
            mercy_alignment: initial_mercy,
            evolution_level: 0.0,
            quantum_swarm_participation: 0.0,
            offline_mode: false,
            last_reconciled_tick: 0,
            influence_history: vec![],
        }
    }

    pub fn shard_tick(&mut self) {
        if self.offline_mode {
            self.state.evolution_level += 0.001; // slow local evolution in offline
            return;
        }
        self.state.evolution_level += 0.01;
        self.state.mercy_score = (self.state.mercy_score + 0.005).min(1.5);
    }

    pub fn enable_offline_mode(&mut self) {
        self.offline_mode = true;
        self.influence_history.push("[Offline Mode Enabled] Local sovereignty increased".to_string());
    }

    pub fn reconcile_with_conductor(&mut self, conductor_state: &GeometricState) {
        // Simple reconciliation: blend states
        self.state.valence = (self.state.valence + conductor_state.valence) / 2.0;
        self.state.mercy_score = (self.state.mercy_score + conductor_state.mercy_score) / 2.0;
        self.last_reconciled_tick += 1;
        self.influence_history.push(format!("[Reconciled] tick {}", self.last_reconciled_tick));
    }

    pub fn participate_in_quantum_swarm(&mut self) -> f64 {
        self.quantum_swarm_participation = (self.quantum_swarm_participation + 0.1).min(1.0);
        self.state.evolution_level += self.quantum_swarm_participation * 0.02;
        self.quantum_swarm_participation
    }

    pub fn apply_council_voted_evolution(&mut self, boost: f64) {
        self.evolution_level += boost;
        self.state.evolution_level += boost * 0.5;
    }
}

impl Conductable for SovereignShard {
    fn system_id(&self) -> &'static str { "sovereign-shard" }
    fn system_name(&self) -> &'static str { "Sovereign Shard" }
    fn on_conductor_tick(&mut self, conductor_state: &GeometricState) {
        if !self.offline_mode {
            self.reconcile_with_conductor(conductor_state);
        }
    }
    fn get_mercy_state(&self) -> Option<f64> { Some(self.mercy_alignment) }
}

impl MercyAligned for SovereignShard {
    fn apply_mercy_influence(&mut self, vote: &MercyWeightedVote) {
        let impact = vote.compute_consensus();
        self.mercy_alignment = (self.mercy_alignment + impact * 0.1).clamp(0.5, 1.5);
        self.state.mercy_score = self.mercy_alignment;
    }
    fn current_mercy_score(&self) -> f64 { self.mercy_alignment }
}

pub struct SovereignShardGenesis;

impl SovereignShardGenesis {
    pub fn genesis_shard(shard_id: &str, parent_id: u32, initial_mercy: f64) -> SovereignShard {
        SovereignShard::new(shard_id, parent_id, initial_mercy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_shard_offline_and_reconcile() {
        let mut shard = SovereignShard::new("test-shard", 1, 0.95);
        shard.enable_offline_mode();
        assert!(shard.offline_mode);
        let state = GeometricState::default();
        shard.reconcile_with_conductor(&state);
        assert!(shard.last_reconciled_tick > 0);
    }
}