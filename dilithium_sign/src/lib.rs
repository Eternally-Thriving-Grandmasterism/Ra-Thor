//! Dilithium Signing — Full Post-Quantum Digital Signatures for Eternal Lattice
//! CRYSTALS-Dilithium Level 3 (128-bit classical security)

use pqcrypto_dilithium::dilithium3;
use pqcrypto_traits::sign::{PublicKey, SecretKey, SignedMessage, VerificationResult};

/// Generate Dilithium keypair
pub fn generate_dilithium_keypair() -> (dilithium3::PublicKey, dilithium3::SecretKey) {
    dilithium3::keypair()
}

/// Sign lattice attestation (e.g., valence proof commitment)
pub fn dilithium_sign_message(
    message: &[u8],
    secret_key: &dilithium3::SecretKey,
) -> dilithium3::SignedMessage {
    dilithium3::sign(message, secret_key)
}

/// Verify Dilithium signature on lattice attestation
pub fn dilithium_verify_signature(
    signed_message: &dilithium3::SignedMessage,
    public_key: &dilithium3::PublicKey,
) -> bool {
    dilithium3::verify(signed_message, public_key).is_ok()
}
