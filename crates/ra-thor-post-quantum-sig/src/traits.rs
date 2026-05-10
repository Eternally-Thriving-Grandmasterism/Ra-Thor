// src/traits.rs
//! Core trait for post-quantum signature algorithms.

use crate::error::PostQuantumError;
use async_trait::async_trait;

#[async_trait]
pub trait PostQuantumSignature: Send + Sync {
    /// Generate a new key pair
    async fn generate_keypair(&self) -> Result<(Vec<u8>, Vec<u8>), PostQuantumError>;

    /// Sign a message
    async fn sign(&self, message: &[u8], private_key: &[u8]) -> Result<Vec<u8>, PostQuantumError>;

    /// Verify a signature
    async fn verify(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool, PostQuantumError>;

    /// Check if the current mercy valence allows this operation
    fn check_mercy_valence(&self, valence: f64) -> Result<(), PostQuantumError> {
        if valence < 0.999 {
            return Err(PostQuantumError::MercyGateRejected { valence });
        }
        Ok(())
    }
}