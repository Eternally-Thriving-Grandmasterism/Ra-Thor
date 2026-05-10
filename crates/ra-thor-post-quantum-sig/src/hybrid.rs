// src/hybrid.rs
//! Hybrid classical + post-quantum signature support (Pass 2).

use crate::error::PostQuantumError;
use crate::traits::PostQuantumSignature;
use async_trait::async_trait;

/// HybridSigner combines a classical signature scheme with a post-quantum one.
pub struct HybridSigner<PQ: PostQuantumSignature> {
    pq_signer: PQ,
}

impl<PQ: PostQuantumSignature> HybridSigner<PQ> {
    pub fn new(pq_signer: PQ) -> Self {
        Self { pq_signer }
    }
}

#[async_trait]
impl<PQ: PostQuantumSignature> PostQuantumSignature for HybridSigner<PQ> {
    async fn generate_keypair(&self) -> Result<(Vec<u8>, Vec<u8>), PostQuantumError> {
        self.pq_signer.generate_keypair().await
    }

    async fn sign(&self, message: &[u8], private_key: &[u8]) -> Result<Vec<u8>, PostQuantumError> {
        // Foundation for hybrid signing. Real implementation will combine
        // classical (e.g. Ed25519) + post-quantum signatures.
        self.pq_signer.sign(message, private_key).await
    }

    async fn verify(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool, PostQuantumError> {
        self.pq_signer.verify(message, signature, public_key).await
    }
}