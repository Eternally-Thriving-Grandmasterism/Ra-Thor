//! Sovereign Channel Encryption v14.0.7+ — Hardened (rand nonces + AAD)
//! Symbiotic encryption support for Self-Evolution Proposals & Powrush actions.

use crate::distributed_mercy_mesh::MercyEvent;
use std::collections::HashMap;

// Add to Cargo.toml:
// aes-gcm = { version = "0.10", features = ["std"] }
// rand = "0.8"
use aes_gcm::{
    aead::{Aead, KeyInit, Payload},
    Aes256Gcm, Key, Nonce,
};
use rand::RngCore;

#[derive(Debug, Clone, PartialEq)]
pub enum ChannelDirection { Outgoing, Incoming, Bidirectional }

#[derive(Debug, Clone, PartialEq)]
pub enum ChannelStatus { Pending, Active, Suspended, Closed }

#[derive(Debug, Clone, PartialEq)]
pub enum EncryptionState { Unencrypted, KeyEstablished, ActiveEncrypted }

#[derive(Debug, Clone)]
pub struct SovereignChannel {
    pub id: String,
    pub from_organism: String,
    pub to_organism: String,
    pub direction: ChannelDirection,
    pub status: ChannelStatus,
    pub encryption_state: EncryptionState,
    pub mercy_score: f64,
    pub last_activity: u64,
    channel_key: Option<[u8; 32]>,
}

impl SovereignChannel {
    pub fn new(from: &str, to: &str, direction: ChannelDirection) -> Self {
        Self {
            id: format!("channel_{}_{}", from, to),
            from_organism: from.to_string(),
            to_organism: to.to_string(),
            direction,
            status: ChannelStatus::Pending,
            encryption_state: EncryptionState::Unencrypted,
            mercy_score: 0.7,
            last_activity: 0,
            channel_key: None,
        }
    }

    pub fn establish_encryption(&mut self, shared_secret: [u8; 32]) {
        self.channel_key = Some(shared_secret);
        self.encryption_state = EncryptionState::KeyEstablished;
    }

    pub fn activate(&mut self) {
        if self.encryption_state == EncryptionState::KeyEstablished {
            self.encryption_state = EncryptionState::ActiveEncrypted;
        }
        self.status = ChannelStatus::Active;
    }

    pub fn close(&mut self) {
        self.status = ChannelStatus::Closed;
        self.encryption_state = EncryptionState::Unencrypted;
        self.channel_key = None;
    }

    /// Cryptographically secure random nonce (hardened).
    fn generate_secure_nonce(&self) -> Nonce<Aes256Gcm> {
        let mut nonce = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce);
        *Nonce::<Aes256Gcm>::from_slice(&nonce)
    }

    /// Hardened encryption with Additional Authenticated Data (AAD) containing mercy metadata.
    pub fn encrypt_with_mercy_aad(&mut self, plaintext: &[u8], mercy_aad: &[u8]) -> Option<Vec<u8>> {
        if self.encryption_state != EncryptionState::ActiveEncrypted { return None; }
        let key = self.channel_key.as_ref()?;
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
        let nonce = self.generate_secure_nonce();

        match cipher.encrypt(&nonce, Payload { msg: plaintext, aad: mercy_aad }) {
            Ok(ct) => {
                let mut result = nonce.to_vec();
                result.extend_from_slice(&ct);
                Some(result)
            }
            Err(_) => None,
        }
    }

    pub fn decrypt_with_mercy_aad(&self, ciphertext: &[u8], mercy_aad: &[u8]) -> Option<Vec<u8>> {
        if self.encryption_state != EncryptionState::ActiveEncrypted || ciphertext.len() < 12 { return None; }
        let key = self.channel_key.as_ref()?;
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
        let (nonce_bytes, data) = ciphertext.split_at(12);
        let nonce = Nonce::<Aes256Gcm>::from_slice(nonce_bytes);
        cipher.decrypt(nonce, Payload { msg: data, aad: mercy_aad }).ok()
    }

    // === Symbiotic helpers ===

    /// Encrypt a Self-Evolution Proposal with mercy AAD.
    pub fn encrypt_self_evolution_proposal(&mut self, proposal_data: &[u8]) -> Option<Vec<u8>> {
        let aad = format!("self-evolution|mercy_score:{:.3}", self.mercy_score).into_bytes();
        self.encrypt_with_mercy_aad(proposal_data, &aad)
    }

    /// Encrypt a Powrush action with mercy AAD.
    pub fn encrypt_powrush_action(&mut self, action_data: &[u8]) -> Option<Vec<u8>> {
        let aad = format!("powrush-action|mercy_score:{:.3}", self.mercy_score).into_bytes();
        self.encrypt_with_mercy_aad(action_data, &aad)
    }
}

pub struct SovereignChannelManager {
    channels: HashMap<String, SovereignChannel>,
}

impl SovereignChannelManager {
    pub fn new() -> Self { Self { channels: HashMap::new() } }

    pub fn create_channel(&mut self, from: &str, to: &str, direction: ChannelDirection) -> &mut SovereignChannel {
        let ch = SovereignChannel::new(from, to, direction);
        let id = ch.id.clone();
        self.channels.insert(id.clone(), ch);
        self.channels.get_mut(&id).unwrap()
    }

    pub fn get_active_encrypted_channels(&self) -> Vec<&SovereignChannel> {
        self.channels.values()
            .filter(|c| c.status == ChannelStatus::Active && c.encryption_state == EncryptionState::ActiveEncrypted)
            .collect()
    }
}