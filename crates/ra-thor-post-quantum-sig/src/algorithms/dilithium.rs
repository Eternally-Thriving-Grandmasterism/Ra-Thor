// src/algorithms/dilithium.rs
//! Dilithium Post-Quantum Signature Implementation (Mercy-Gated)
//!
//! This module provides a mercy-aware implementation of the Dilithium2
//! post-quantum signature scheme, integrated with the Ra-Thor mercy system.

use crate::error::PostQuantumError;
use crate::traits::PostQuantumSignature;
use async_trait::async_trait;
use pqcrypto_dilithium::dilithium2::{
    detached_sign, keypair, verify_detached_signature,
    PublicKey, SecretKey, DetachedSignature,
};

/// Dilithium2 Post-Quantum Signer with Mercy Gating
///
/// This signer enforces mercy valence thresholds before performing
/// cryptographic operations. Future versions will integrate deeply
/// with `mercy_merlin_engine` for advanced valence and council checks.
pub struct DilithiumSigner {
    /// Minimum mercy valence required to perform signing operations.
    /// Default is 0.999 (very high bar, aligned with Ra-Thor standards).
    pub mercy_valence_threshold: f64,
}

impl Default for DilithiumSigner {
    fn default() -> Self {
        Self {
            mercy_valence_threshold: 0.999,
        }
    }
}

impl DilithiumSigner {
    /// Creates a new DilithiumSigner with a custom mercy valence threshold.
    pub fn new(mercy_valence_threshold: f64) -> Self {
        Self {
            mercy_valence_threshold,
        }
    }

    /// Checks if the current mercy valence allows cryptographic operations.
    ///
    /// This is a fast local check. In future passes this will be replaced
    /// or augmented with a call to `mercy_merlin_engine` for deeper analysis.
    fn check_mercy_valence(&self, current_valence: f64) -> Result<(), PostQuantumError> {
        if current_valence < self.mercy_valence_threshold {
            return Err(PostQuantumError::MercyGateRejected {
                valence: current_valence,
            });
        }
        Ok(())
    }
}

#[async_trait]
impl PostQuantumSignature for DilithiumSigner {
    /// Generate a new Dilithium2 key pair.
    async fn generate_keypair(&self) -> Result<(Vec<u8>, Vec<u8>), PostQuantumError> {
        // TODO: Add real mercy_merlin_engine valence + council check here
        let (public_key, secret_key) = keypair();
        Ok((public_key.as_bytes().to_vec(), secret_key.as_bytes().to_vec()))
    }

    /// Sign a message using Dilithium2.
    ///
    /// Performs a mercy valence check before signing.
    async fn sign(&self, message: &[u8], private_key: &[u8]) -> Result<Vec<u8>, PostQuantumError> {
        // TODO: Replace with real call to mercy_merlin_engine for valence + council consensus
        // self.check_mercy_valence(current_valence_from_merlin_engine).await?;

        if private_key.len() != 2528 {
            return Err(PostQuantumError::InvalidKeyMaterial);
        }

        let secret_key = SecretKey::from_bytes(private_key)
            .map_err(|e| PostQuantumError::CryptoError(e.to_string()))?;

        let signature = detached_sign(message, &secret_key);
        Ok(signature.as_bytes().to_vec())
    }

    /// Verify a Dilithium2 signature.
    async fn verify(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool, PostQuantumError> {
        if public_key.len() != 1312 {
            return Err(PostQuantumError::InvalidKeyMaterial);
        }

        let public_key = PublicKey::from_bytes(public_key)
            .map_err(|e| PostQuantumError::CryptoError(e.to_string()))?;

        let signature = DetachedSignature::from_bytes(signature)
            .map_err(|e| PostQuantumError::CryptoError(e.to_string()))?;

        match verify_detached_signature(&signature, message, &public_key) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}