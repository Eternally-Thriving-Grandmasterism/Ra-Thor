// src/algorithms/dilithium.rs
//! Dilithium Post-Quantum Signature Implementation (Mercy-Gated)
//!
//! This module provides a mercy-aware Dilithium2 implementation integrated
//! with the Ra-Thor mercy system and TOLC mathematics.

use crate::error::PostQuantumError;
use crate::traits::PostQuantumSignature;
use async_trait::async_trait;
use pqcrypto_dilithium::dilithium2::{
    detached_sign, keypair, verify_detached_signature,
    PublicKey, SecretKey, DetachedSignature,
};

/// Dilithium2 Post-Quantum Signer with Mercy Gating
pub struct DilithiumSigner {
    /// Minimum mercy valence required to perform signing operations.
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

    /// Checks mercy valence before performing sensitive cryptographic operations.
    ///
    /// In future passes, this will make a real call to `mercy_merlin_engine`
    /// to get the current valence and council consensus.
    async fn ensure_mercy_allowed(&self) -> Result<(), PostQuantumError> {
        // TODO: Replace this stub with real call to mercy_merlin_engine
        // Example future call:
        // let current_valence = mercy_merlin_engine::get_current_valence().await?;
        // if current_valence < self.mercy_valence_threshold {
        //     return Err(PostQuantumError::MercyGateRejected { valence: current_valence });
        // }

        // For now, we assume the caller has already verified mercy conditions
        // or we perform a basic local check.
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
        // Future: Call mercy_merlin_engine here for valence + council check
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