// src/error.rs
//! Mercy-gated error types for post-quantum signature operations.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PostQuantumError {
    #[error("Mercy gate rejected operation: valence too low ({valence})")]
    MercyGateRejected { valence: f64 },

    #[error("Council consensus failed for cryptographic operation")]
    CouncilConsensusFailed,

    #[error("Invalid key material")]
    InvalidKeyMaterial,

    #[error("Signature verification failed")]
    VerificationFailed,

    #[error("Key generation failed")]
    KeyGenerationFailed,

    #[error("TOLC integration error: {0}")]
    TOLCError(String),

    #[error("Internal cryptographic error: {0}")]
    CryptoError(String),
}