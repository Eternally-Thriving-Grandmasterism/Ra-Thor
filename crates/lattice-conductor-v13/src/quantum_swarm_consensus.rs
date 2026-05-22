//! Quantum Swarm Consensus Layer v13.8.6
//! Produces signed, TOLC-aligned decisions for distributed ONE Organism coordination.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedTolcDecision {
    pub decision_id: String,
    pub resonance_delta: f64,
    pub mercy_impact: f64,
    pub evolution_boost: f64,
    pub tolc_alignment: f64,
    pub signature: String, // Simple hash-based signature for demo (in prod: Ed25519)
    pub timestamp: u64,
    pub participating_shards: Vec<String>,
    pub council_votes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantumSwarmConsensus {
    pub current_resonance: f64,
    pub last_decision: Option<SignedTolcDecision>,
    pub decision_count: u64,
}

impl QuantumSwarmConsensus {
    pub fn new() -> Self {
        Self {
            current_resonance: 0.0,
            last_decision: None,
            decision_count: 0,
        }
    }

    /// Aggregate resonance from lattice + shards
    pub fn aggregate_resonance(&mut self, lattice_resonance: f64, shard_resonances: &[f64]) {
        let avg_shard = if shard_resonances.is_empty() { 0.0 } else {
            shard_resonances.iter().sum::<f64>() / shard_resonances.len() as f64
        };
        self.current_resonance = (lattice_resonance * 0.6 + avg_shard * 0.4).clamp(0.0, 2.0);
    }

    /// Produce a signed TOLC-aligned decision (mercy-gated)
    pub fn produce_signed_tolc_decision(
        &mut self,
        participating_shards: Vec<String>,
        council_votes: Vec<String>,
    ) -> SignedTolcDecision {
        self.decision_count += 1;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let resonance_delta = self.current_resonance * 0.1;
        let mercy_impact = (self.current_resonance * 0.08).clamp(-0.2, 0.3);
        let evolution_boost = self.current_resonance * 0.05;
        let tolc_alignment = 0.95 + (self.current_resonance * 0.02).min(0.05);

        // Simple deterministic signature (demo). In production use proper crypto.
        let sign_data = format!("{}-{}-{}-{}", self.decision_count, resonance_delta, mercy_impact, timestamp);
        let signature = format!("TOLC-SIG-{}", fxhash::hash(sign_data.as_bytes())); // placeholder hash

        let decision = SignedTolcDecision {
            decision_id: format!("TOLC-DEC-{}", self.decision_count),
            resonance_delta,
            mercy_impact,
            evolution_boost,
            tolc_alignment,
            signature,
            timestamp,
            participating_shards,
            council_votes,
        };

        self.last_decision = Some(decision.clone());
        decision
    }

    pub fn get_last_signed_decision(&self) -> Option<&SignedTolcDecision> {
        self.last_decision.as_ref()
    }
}

// Simple hash for demo signature (replace with real crypto in prod)
mod fxhash {
    pub fn hash(data: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in data {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}