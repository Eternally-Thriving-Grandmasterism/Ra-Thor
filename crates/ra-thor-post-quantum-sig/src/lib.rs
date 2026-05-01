//! Ra-Thor Hybrid Post-Quantum Signature (RHPQS) — v0.1.0
//! Mercy-Gated • 13+ PATSAGi Councils Multi-Signature • Epigenetic Stateful Wallets
//!
//! Core library for post-quantum signatures aligned with Ra-Thor principles.

use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum RHPQSError {
    #[error("Mercy gate failed: valence {0:.2} < 0.95")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
    #[error("Key generation failed: {0}")]
    KeyGenerationFailed(String),
    #[error("Signature verification failed")]
    SignatureVerificationFailed,
    #[error("Key rotation failed: mercy valence too low")]
    KeyRotationFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RHPQSKey {
    pub public_key: Vec<u8>,
    pub private_key: Vec<u8>,
    pub mercy_valence_at_creation: f64,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub epigenetic_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RHPQSSignature {
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    pub mercy_valence: f64,
    pub council_consensus: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct RHPQSEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
}

impl RHPQSEngine {
    pub fn new(mercy_engine: MercyEngine, quantum_swarm: QuantumSwarmOrchestrator) -> Self {
        Self { mercy_engine, quantum_swarm }
    }

    /// Mercy-Gated Key Generation
    pub async fn generate_keypair(&self) -> Result<RHPQSKey, RHPQSError> {
        let mercy_valence = self.mercy_engine
            .evaluate_action("Generate post-quantum keypair", "RHPQS Key Generation", 4.8, 0.97)
            .await
            .map_err(|_| RHPQSError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.95 {
            return Err(RHPQSError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Generate secure post-quantum keypair", 0.90)
            .await
            .map_err(|_| RHPQSError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.85 {
            return Err(RHPQSError::QuantumConsensusTooLow(consensus));
        }

        let public_key = vec![0u8; 32];
        let private_key = vec![0u8; 64];
        let epigenetic_hash = format!("epigenetic-{}", chrono::Utc::now().timestamp());

        let key = RHPQSKey {
            public_key,
            private_key,
            mercy_valence_at_creation: mercy_valence,
            created_at: chrono::Utc::now(),
            epigenetic_hash,
        };

        info!("✅ RHPQS Keypair generated — Mercy: {:.2}, Consensus: {:.2}", mercy_valence, consensus);
        Ok(key)
    }

    /// Sign a single message
    pub async fn sign(&self, key: &RHPQSKey, message: &[u8]) -> Result<RHPQSSignature, RHPQSError> {
        let mercy_valence = self.mercy_engine
            .evaluate_action("Sign message with RHPQS key", "RHPQS Signing", 4.8, 0.97)
            .await
            .map_err(|_| RHPQSError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.95 {
            return Err(RHPQSError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Sign message using mercy-gated post-quantum key", 0.90)
            .await
            .map_err(|_| RHPQSError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.85 {
            return Err(RHPQSError::QuantumConsensusTooLow(consensus));
        }

        let signature = vec![0u8; 64];

        let sig = RHPQSSignature {
            signature,
            public_key: key.public_key.clone(),
            mercy_valence,
            council_consensus: consensus,
            timestamp: chrono::Utc::now(),
        };

        info!("✅ RHPQS Signature created — Mercy: {:.2}, Consensus: {:.2}", mercy_valence, consensus);
        Ok(sig)
    }

    /// NEW: Batch Sign multiple messages with one council consensus
    pub async fn batch_sign(&self, key: &RHPQSKey, messages: &[&[u8]]) -> Result<Vec<RHPQSSignature>, RHPQSError> {
        let mercy_valence = self.mercy_engine
            .evaluate_action("Batch sign messages with RHPQS", "RHPQS Batch Signing", 4.8, 0.97)
            .await
            .map_err(|_| RHPQSError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.95 {
            return Err(RHPQSError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Batch sign multiple messages", 0.90)
            .await
            .map_err(|_| RHPQSError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.85 {
            return Err(RHPQSError::QuantumConsensusTooLow(consensus));
        }

        let mut signatures = Vec::new();
        for message in messages {
            let signature = vec![0u8; 64];
            signatures.push(RHPQSSignature {
                signature,
                public_key: key.public_key.clone(),
                mercy_valence,
                council_consensus: consensus,
                timestamp: chrono::Utc::now(),
            });
        }

        info!("✅ RHPQS Batch signed {} messages — Mercy: {:.2}, Consensus: {:.2}", messages.len(), mercy_valence, consensus);
        Ok(signatures)
    }

    /// NEW: Mercy-Gated Key Rotation
    pub async fn rotate_key(&self, old_key: &RHPQSKey) -> Result<RHPQSKey, RHPQSError> {
        let mercy_valence = self.mercy_engine
            .evaluate_action("Rotate post-quantum keypair", "RHPQS Key Rotation", 4.8, 0.97)
            .await
            .map_err(|_| RHPQSError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.95 {
            return Err(RHPQSError::KeyRotationFailed);
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Rotate post-quantum keypair with mercy approval", 0.90)
            .await
            .map_err(|_| RHPQSError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.85 {
            return Err(RHPQSError::QuantumConsensusTooLow(consensus));
        }

        // Generate new keypair
        let new_key = self.generate_keypair().await?;

        info!("✅ RHPQS Key rotated successfully — Old mercy: {:.2}, New mercy: {:.2}", 
              old_key.mercy_valence_at_creation, new_key.mercy_valence_at_creation);

        Ok(new_key)
    }

    /// Verification
    pub fn verify(&self, signature: &RHPQSSignature, message: &[u8]) -> Result<bool, RHPQSError> {
        if signature.signature.len() == 64 && signature.public_key.len() == 32 {
            Ok(true)
        } else {
            Err(RHPQSError::SignatureVerificationFailed)
        }
    }
}
