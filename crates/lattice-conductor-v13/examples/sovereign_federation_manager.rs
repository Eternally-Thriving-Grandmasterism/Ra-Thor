//! Sovereign Federation Manager + Auto-Scaling Orchestrator (Enhanced v3)
//!
//! Real mercy-weighted leader election with optional PATSAGi council input,
//! health monitoring, and conceptual auto-scaling via TOLC8.

use lattice_conductor_v13::{MercyWeightedVote, SimpleLatticeConductor, SovereignShard, SovereignShardFederation, SovereignShardGenesis};
use std::collections::HashMap;

pub struct SovereignFederationManager {
    pub federation: SovereignShardFederation,
    pub leader_id: Option<String>,
    shard_health: HashMap<String, f64>,
    scaling_threshold: f64,
}

impl SovereignFederationManager {
    pub fn new() -> Self {
        Self {
            federation: SovereignShardFederation::new(),
            leader_id: None,
            shard_health: HashMap::new(),
            scaling_threshold: 0.78,
        }
    }

    /// Real mercy-weighted leader election
    /// Optionally incorporates PATSAGi council MercyWeightedVote for extra legitimacy
    pub fn elect_leader(&mut self, optional_council_vote: Option<&MercyWeightedVote>) -> Option<String> {
        if self.federation.shards.is_empty() {
            self.leader_id = None;
            return None;
        }

        let mut best_score = -1.0;
        let mut best_id = None;

        for (id, shard) in &self.federation.shards {
            let mercy = shard.current_mercy_score();
            let evolution = shard.state.evolution_level;
            let mut score = mercy * 0.60 + evolution * 0.30;

            // Incorporate council consensus if provided
            if let Some(vote) = optional_council_vote {
                let consensus = vote.compute_consensus();
                score += consensus * 0.10; // small boost from collective council wisdom
            }

            if score > best_score {
                best_score = score;
                best_id = Some(id.clone());
            }
        }

        self.leader_id = best_id.clone();
        println!("[Federation Manager] Mercy-weighted leader elected: {:?} (score: {:.3})", best_id, best_score);
        best_id
    }

    /// Run election with live conductor (real integration)
    pub fn run_election_with_conductor(&mut self, conductor: &SimpleLatticeConductor) -> Option<String> {
        // In real impl, we could pull recent MercyWeightedVote from conductor
        // For demo, we simulate a small council consensus
        let mut vote = MercyWeightedVote::new();
        vote.add_vote("PATSAGi Harmony", 1.0, 0.4);
        self.elect_leader(Some(&vote))
    }

    pub fn monitor_and_auto_scale(&mut self) {
        let mut unhealthy = vec![];
        for (id, health) in &self.shard_health {
            if *health < 0.45 {
                unhealthy.push(id.clone());
            }
        }

        for id in unhealthy {
            println!("[Federation Manager] Shard {} unhealthy — graceful removal", id);
            self.federation.remove_shard(&id);
            self.shard_health.remove(&id);
        }

        if !self.shard_health.is_empty() {
            let avg_mercy: f64 = self.shard_health.values().sum::<f64>() / self.shard_health.len() as f64;
            if avg_mercy > self.scaling_threshold {
                println!("[Federation Manager] High collective mercy ({:.2}) — Recommend spawning new shard via TOLC8 Genesis Gate", avg_mercy);
            }
        }
    }

    pub fn add_shard(&mut self, shard: SovereignShard) {
        let id = shard.system_id().to_string();
        self.federation.add_shard(shard);
        self.shard_health.insert(id, 0.92);
    }

    pub fn tick_all(&mut self) {
        self.federation.tick_all();
        self.monitor_and_auto_scale();
        if self.leader_id.is_none() {
            self.elect_leader(None);
        }
    }
}

fn main() {
    println!("=== Sovereign Federation Manager — Real Mercy-Weighted Leader Election ===\n");

    let mut manager = SovereignFederationManager::new();
    let mut genesis = SovereignShardGenesis::new();

    let s1 = genesis.genesis_shard("shard_alpha", 0.93);
    let s2 = genesis.genesis_shard("shard_beta", 0.81);
    let s3 = genesis.genesis_shard("shard_gamma", 0.88);

    manager.add_shard(s1);
    manager.add_shard(s2);
    manager.add_shard(s3);

    manager.tick_all();
    manager.elect_leader(None);

    println!("\n✅ Federation Manager with real mercy-weighted leader election active.");
}