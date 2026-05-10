// src/algorithms/dilithium.rs
//! Dilithium Post-Quantum Signature Implementation (Mercy-Gated)

use crate::error::PostQuantumError;
use crate::traits::PostQuantumSignature;
use async_trait::async_trait;
use pqcrypto_dilithium::dilithium2::{
    detached_sign, keypair, verify_detached_signature,
    PublicKey, SecretKey, DetachedSignature,
};

pub struct DilithiumSigner {
    pub mercy_valence_threshold: f64,
}

impl Default for DilithiumSigner {
    fn default() -> Self {
        Self {
            mercy_valence_threshold: 0.999,
        }
    }
}

#[async_trait]
impl PostQuantumSignature for DilithiumSigner {
    async fn generate_keypair(&self) -> Result<(Vec<u8>, Vec<u8>), PostQuantumError> {
        let (public_key, secret_key) = keypair();
        Ok((public_key.as_bytes().to_vec(), secret_key.as_bytes().to_vec()))
    }

    async fn sign(&self, message: &[u8], private_key: &[u8]) -> Result<Vec<u8>, PostQuantumError> {
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