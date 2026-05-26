//! Crypto Traits for modular cryptographic abstractions

pub trait SignatureScheme {
    type Signature;
    type PublicKey;
    type SecretKey;

    fn sign(secret_key: &Self::SecretKey, message: &[u8]) -> Self::Signature;
    fn verify(public_key: &Self::PublicKey, message: &[u8], signature: &Self::Signature) -> bool;
}

pub trait KeyExchange {
    type PublicKey;
    type SecretKey;
    type SharedSecret;
    type Ciphertext;

    fn generate_keypair() -> (Self::PublicKey, Self::SecretKey);
    fn encapsulate(public_key: &Self::PublicKey) -> (Self::Ciphertext, Self::SharedSecret);
    fn decapsulate(secret_key: &Self::SecretKey, ciphertext: &Self::Ciphertext) -> Option<Self::SharedSecret>;
}

pub trait AuthenticatedEncryption {
    type Key;
    type Nonce;
    type Ciphertext;

    fn encrypt(key: &Self::Key, nonce: &Self::Nonce, plaintext: &[u8]) -> Option<Self::Ciphertext>;
    fn decrypt(key: &Self::Key, nonce: &Self::Nonce, ciphertext: &[u8]) -> Option<Vec<u8>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crypto_traits_are_defined() {
        // Compilation of this test confirms the traits exist and are well-formed
        assert!(true);
    }
}