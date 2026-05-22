//! Sovereign Shard Genesis v13.8.7
//!
//! Self-sovereign, mercy-gated shards that participate in the ONE Organism
//! via Conductable + MercyAligned traits and ConductorRegistry blessing.
//! Now with full offline reconciliation, multi-shard federation, and persistent storage.

use lattice_conductor_v13::{
    Conductable, ConductorRegistry, GeometricState, MercyAligned, MercyWeightedVote, SystemBlessing,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

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
    pub offline_mode: bool,
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
            offline_mode: false,
        }
    }

    pub fn enable_offline_mode(&mut self) {
        self.offline_mode = true;
    }

    pub fn disable_offline_mode(&mut self) {
        self.offline_mode = false;
    }

    pub fn is_offline(&self) -> bool {
        self.offline_mode
    }

    pub fn shard_tick(&mut self) {
        self.local_tick_count += 1;
        if self.offline_mode {
            // Local sovereignty: slower but independent evolution
            self.state.evolution_level += 0.003;
            self.state.mercy_score = (self.state.mercy_score + 0.001).min(1.15);
        } else {
            self.state.evolution_level += 0.005;
            self.state.mercy_score = (self.state.mercy_score + 0.002).min(1.2);
        }
    }

    pub fn apply_council_voted_evolution(&mut self, boost: f64) {
        self.evolution_level += boost;
        self.state.evolution_level += boost * 0.5;
    }

    /// Full offline reconciliation protocol: mercy-priority blend when coming back online
    pub fn reconcile_with_conductor(&mut self, conductor_state: &GeometricState, conductor_mercy: f64) {
        if self.offline_mode {
            self.disable_offline_mode();
        }
        // Mercy-first reconciliation (higher weight on global mercy for coherence)
        self.state.mercy_score = (self.state.mercy_score * 0.55 + conductor_mercy * 0.45).clamp(0.5, 1.5);
        self.state.valence = (self.state.valence * 0.7 + conductor_state.valence * 0.3).clamp(0.5, 1.8);
        self.evolution_level = (self.evolution_level + conductor_state.evolution_level * 0.2).max(self.evolution_level);
        self.state.tolc_alignment = (self.state.tolc_alignment + 0.01).min(1.1);
    }

    /// Persistent storage
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        serde_json::from_str(&contents)
    }
}

impl Conductable for SovereignShard {
    fn system_id(&self) -> &'static str { "sovereign_shard" }
    fn system_name(&self) -> &'static str { "Sovereign Shard" }
    fn on_conductor_tick(&mut self, conductor_state: &GeometricState) {
        if !self.offline_mode {
            self.state.valence = (self.state.valence + conductor_state.valence * 0.1).clamp(0.5, 1.8);
        }
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

/// Multi-shard federation manager for collective sovereignty
#[derive(Debug, Default)]
pub struct SovereignShardFederation {
    pub shards: HashMap<String, SovereignShard>,
}

impl SovereignShardFederation {
    pub fn new() -> Self {
        Self { shards: HashMap::new() }
    }

    pub fn add_shard(&mut self, shard: SovereignShard) {
        self.shards.insert(shard.shard_id.clone(), shard);
    }

    pub fn remove_shard(&mut self, shard_id: &str) -> Option<SovereignShard> {
        self.shards.remove(shard_id)
    }

    pub fn get_shard(&self, shard_id: &str) -> Option<&SovereignShard> {
        self.shards.get(shard_id)
    }

    pub fn get_shard_mut(&mut self, shard_id: &str) -> Option<&mut SovereignShard> {
        self.shards.get_mut(shard_id)
    }

    pub fn tick_all(&mut self) {
        for shard in self.shards.values_mut() {
            shard.shard_tick();
        }
    }

    pub fn reconcile_all_with_conductor(&mut self, conductor_state: &GeometricState, conductor_mercy: f64) {
        for shard in self.shards.values_mut() {
            shard.reconcile_with_conductor(conductor_state, conductor_mercy);
        }
    }

    pub fn collective_mercy_score(&self) -> f64 {
        if self.shards.is_empty() { return 0.0; }
        self.shards.values().map(|s| s.mercy_alignment).sum::<f64>() / self.shards.len() as f64
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

    #[test]
    fn test_offline_mode_and_reconciliation() {
        let mut shard = SovereignShard::new("shard-beta-002", "Beta Shard", 0);
        shard.enable_offline_mode();
        assert!(shard.is_offline());
        shard.shard_tick(); // local evolution
        let conductor_state = GeometricState { valence: 1.2, mercy_score: 1.0, tolc_alignment: 1.0, evolution_level: 0.5 };
        shard.reconcile_with_conductor(&conductor_state, 1.0);
        assert!(!shard.is_offline());
        assert!(shard.state.mercy_score > 0.9);
    }

    #[test]
    fn test_federation_and_collective() {
        let mut fed = SovereignShardFederation::new();
        let s1 = SovereignShard::new("shard-1", "Shard One", 0);
        let s2 = SovereignShard::new("shard-2", "Shard Two", 0);
        fed.add_shard(s1);
        fed.add_shard(s2);
        fed.tick_all();
        assert_eq!(fed.shards.len(), 2);
        let avg = fed.collective_mercy_score();
        assert!(avg > 0.9);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let shard = SovereignShard::new("shard-persist-003", "Persist Shard", 0);
        let path = "/tmp/test_shard_persist.json";
        let _ = shard.save_to_file(path);
        let loaded = SovereignShard::load_from_file(path).expect("load failed");
        assert_eq!(loaded.shard_id, "shard-persist-003");
        // cleanup
        let _ = std::fs::remove_file(path);
    }
}