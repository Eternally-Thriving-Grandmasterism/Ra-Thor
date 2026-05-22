//! Quantum Swarm Consensus Layer for ONE Organism
//! Produces signed, TOLC-aligned decisions using real Ed25519 cryptography.

use serde::{Deserialize, Serialize};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey, Signature};
use rand_core::OsRng;
use std::collections::HashMap;
use chrono::Utc;
use hex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedTolcDecision {
    pub decision_id: String,
    pub resonance_delta: f64,
    pub mercy_impact: f64,
    pub evolution_boost: f64,
    pub tolc_alignment: f64,
    pub signature: String, // hex encoded
    pub timestamp: u64,
    pub participating_shards: Vec<String>,
    pub council_votes: HashMap<String, f64>,
}

pub struct QuantumSwarmConsensus {
    pub resonance: f64,
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
}

impl QuantumSwarmConsensus {
    pub fn new() -> Self {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        Self {
            resonance: 0.0,
            signing_key,
            verifying_key,
        }
    }

    pub fn aggregate_resonance(&mut self, lattice_resonance: f64, shard_resonance: f64) {
        self.resonance = (lattice_resonance * 0.6 + shard_resonance * 0.4).clamp(0.0, 1.5);
    }

    pub fn produce_signed_tolc_decision(
        &self,
        resonance_delta: f64,
        mercy_impact: f64,
        evolution_boost: f64,
        tolc_alignment: f64,
        participating_shards: Vec<String>,
        council_votes: HashMap<String, f64>,
    ) -> SignedTolcDecision {
        let now = Utc::now();
        let decision_id = format!("tolc-decision-{}", now.timestamp());
        let timestamp = now.timestamp() as u64;

        // Create a deterministic message to sign
        let message = format!(
            "{}|{}|{}|{}|{}|{:?}|{:?}",
            decision_id, resonance_delta, mercy_impact, evolution_boost, tolc_alignment, participating_shards, council_votes
        );

        let signature: Signature = self.signing_key.sign(message.as_bytes());
        let signature_hex = hex::encode(signature.to_bytes());

        SignedTolcDecision {
            decision_id,
            resonance_delta,
            mercy_impact,
            evolution_boost,
            tolc_alignment,
            signature: signature_hex,
            timestamp,
            participating_shards,
            council_votes,
        }
    }

    pub fn get_verifying_key(&self) -> VerifyingKey {
        self.verifying_key
    }
}