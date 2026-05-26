//! Crypto Traits for modular cryptographic abstractions
//! Enables swapping between classical and post-quantum implementations.

/// A generic signature scheme trait.
/// This allows different signature algorithms (classical or post-quantum)
/// to be used interchangeably.
pub trait SignatureScheme {
    type Signature;
    type PublicKey;
    type SecretKey;

    /// Sign a message using the secret key.
    fn sign(secret_key: &Self::SecretKey, message: &[u8]) -> Self::Signature;

    /// Verify a signature using the public key.
    fn verify(public_key: &Self::PublicKey, message: &[u8], signature: &Self::Signature) -> bool;
}

/// A trait for key exchange / encapsulation mechanisms (e.g., classical ECDH or Kyber KEM).
pub trait KeyExchange {
    type PublicKey;
    type SecretKey;
    type SharedSecret;
    type Ciphertext;

    fn generate_keypair() -> (Self::PublicKey, Self::SecretKey);

    /// Encapsulate a shared secret (used by the initiator).
    fn encapsulate(public_key: &Self::PublicKey) -> (Self::Ciphertext, Self::SharedSecret);

    /// Decapsulate a shared secret (used by the receiver).
    fn decapsulate(secret_key: &Self::SecretKey, ciphertext: &Self::Ciphertext) -> Option<Self::SharedSecret>;
}

/// A trait for authenticated encryption (e.g., AES-GCM or future post-quantum AEAD).
pub trait AuthenticatedEncryption {
    type Key;
    type Nonce;
    type Ciphertext;

    fn encrypt(key: &Self::Key, nonce: &Self::Nonce, plaintext: &[u8]) -> Option<Self::Ciphertext>;
    fn decrypt(key: &Self::Key, nonce: &Self::Nonce, ciphertext: &[u8]) -> Option<Vec<u8>>;
}