//! Falcon Signing — Full Post-Quantum Digital Signatures (Diversified Lattice)
//! CRYSTALS-Falcon Level 5 (256-bit classical security)

use pqcrypto_falcon::falcon1024;
use pqcrypto_traits::sign::{PublicKey, SecretKey, SignedMessage};

/// Generate Falcon keypair (Level 5)
pub fn generate_falcon_keypair() -> (falcon1024::PublicKey, falcon1024::SecretKey) {
    falcon1024::keypair()
}

/// Sign lattice attestation with Falcon
pub fn falcon_sign_message(
    message: &[u8],
    secret_key: &falcon1024::SecretKey,
) -> falcon1024::SignedMessage {
    falcon1024::sign(message, secret_key)
}

/// Verify Falcon signature on lattice attestation
pub fn falcon_verify_signature(
    signed_message: &falcon1024::SignedMessage,
    public_key: &falcon1024::PublicKey,
) -> bool {
    falcon1024::verify(signed_message, public_key).is_ok()
}
