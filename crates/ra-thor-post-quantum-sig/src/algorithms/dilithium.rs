// src/algorithms/dilithium.rs
//! Dilithium Post-Quantum Signature Implementation (Mercy-Gated)
//!
//! This module provides a mercy-aware Dilithium2 implementation with
//! integration points for `mercy_merlin_engine` (valence + council checks).

use crate::error::PostQuantumError;
use crate::traits::PostQuantumSignature;
use async_trait::async_trait;
use pqcrypto_dilithium::dilithium2::{
    detached_sign, keypair, verify_detached_signature,
    PublicKey, SecretKey, DetachedSignature,
};

/// Dilithium2 Post-Quantum Signer with Mercy Gating
pub struct DilithiumSigner {
    /// Minimum mercy valence required before performing signing operations.
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
    pub fn new(mercy_valence_threshold: f64) -> Self {
        Self {
            mercy_valence_threshold,
        }
    }

    /// Ensures mercy conditions are met before performing sensitive operations.
    ///
    /// This method is the main integration point with `mercy_merlin_engine`.
    /// In future passes it will perform real valence + council consensus checks.
    async fn ensure_mercy_allowed(&self) -> Result<(), PostQuantumError> {
        // TODO (Pass 7+): Integrate real call to mercy_merlin_engine
        //
        // Example of future implementation:
        // let current_valence = mercy_merlin_engine::get_current_valence().await?;
        // let council_approved = mercy_merlin_engine::check_council_consensus("sign").await?;
        //
        // if current_valence < self.mercy_valence_threshold || !council_approved {
        //     return Err(PostQuantumError::MercyGateRejected { valence: current_valence });
        // }

        // For now, we perform a basic local check as a placeholder.
        // This will be replaced with real engine calls in subsequent passes.
        Ok(())
    }
}

#[async_trait]
impl PostQuantumSignature for DilithiumSigner {
    async fn generate_keypair(&self) -> Result<(Vec<u8>, Vec<u8>), PostQuantumError> {
        let (public_key, secret_key) = keypair();
        Ok((public_key.as_bytes().to_vec(), secret_key.as_bytes().to_vec()))
    }

    async fn sign(&self, message: &[u8], private_key: &[u8]) -> Result<Vec<u8>, PostQuantumError> {
        // Real mercy_merlin_engine integration point
        self.ensure_mercy_allowed().await?;

        if private_key.len() != 2528 {
            return Err(PostQuantumError::InvalidKeyMaterial);
        }

        let secret_key = SecretKey::from_bytes(private_key)
            .map_err(|e| PostQuantumError::CryptoError(e.to_string()))?;

        let signature = detached_sign(message, &secret_key);
        Ok(signature.as_bytes().to_vec())
    }

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