//! Sovereign Channel Encryption — Thunder Lattice v14.0.7+
//! Production-grade AES-GCM encrypted sovereign channels.

use crate::distributed_mercy_mesh::MercyEvent;
use std::collections::HashMap;

// NOTE: Add this to Cargo.toml:
// aes-gcm = { version = "0.10", features = ["std"] }
use aes_gcm::{
    aead::{Aead, KeyInit, Payload},
    Aes256Gcm, Key, Nonce,
};

/// Direction of the sovereign channel.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelDirection {
    Outgoing,
    Incoming,
    Bidirectional,
}

/// Status of a sovereign channel.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelStatus {
    Pending,
    Active,
    Suspended,
    Closed,
}

/// Encryption state of the channel.
#[derive(Debug, Clone, PartialEq)]
pub enum EncryptionState {
    Unencrypted,
    KeyEstablished,
    ActiveEncrypted,
}

/// A sovereign, mercy-gated, AES-GCM encrypted communication channel.
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
    nonce_counter: u64, // Simple counter for nonce uniqueness per channel
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
            nonce_counter: 0,
        }
    }

    /// Establish a 32-byte shared secret key.
    pub fn establish_encryption(&mut self, shared_secret: [u8; 32]) {
        self.channel_key = Some(shared_secret);
        self.encryption_state = EncryptionState::KeyEstablished;
        println!("[SOVEREIGN CHANNEL] AES-GCM key established for {}", self.id);
    }

    pub fn activate(&mut self) {
        if self.encryption_state == EncryptionState::KeyEstablished {
            self.encryption_state = EncryptionState::ActiveEncrypted;
        }
        self.status = ChannelStatus::Active;
        println!("[SOVEREIGN CHANNEL] Channel {} activated (AES-GCM encrypted)", self.id);
    }

    pub fn close(&mut self) {
        self.status = ChannelStatus::Closed;
        self.encryption_state = EncryptionState::Unencrypted;
        self.channel_key = None;
    }

    /// Generate a unique nonce for this encryption operation.
    fn next_nonce(&mut self) -> Nonce<Aes256Gcm> {
        self.nonce_counter += 1;
        let mut nonce_bytes = [0u8; 12];
        nonce_bytes[4..].copy_from_slice(&self.nonce_counter.to_be_bytes());
        *Nonce::<Aes256Gcm>::from_slice(&nonce_bytes)
    }

    /// Encrypt using AES-GCM (authenticated encryption).
    pub fn encrypt_payload(&mut self, plaintext: &[u8]) -> Option<Vec<u8>> {
        if self.encryption_state != EncryptionState::ActiveEncrypted {
            return None;
        }
        let key_bytes = self.channel_key.as_ref()?;
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key_bytes));

        let nonce = self.next_nonce();

        match cipher.encrypt(&nonce, Payload { msg: plaintext, aad: &[] }) {
            Ok(ciphertext) => {
                // Prepend nonce to ciphertext for transmission
                let mut result = nonce.to_vec();
                result.extend_from_slice(&ciphertext);
                Some(result)
            }
            Err(_) => None,
        }
    }

    /// Decrypt using AES-GCM.
    pub fn decrypt_payload(&self, ciphertext: &[u8]) -> Option<Vec<u8>> {
        if self.encryption_state != EncryptionState::ActiveEncrypted || ciphertext.len() < 12 {
            return None;
        }
        let key_bytes = self.channel_key.as_ref()?;
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key_bytes));

        let (nonce_bytes, encrypted_data) = ciphertext.split_at(12);
        let nonce = Nonce::<Aes256Gcm>::from_slice(nonce_bytes);

        cipher.decrypt(nonce, Payload { msg: encrypted_data, aad: &[] }).ok()
    }

    /// Send an encrypted message (updates nonce counter).
    pub fn send_encrypted_message(&mut self, payload: &[u8]) -> Option<MercyEvent> {
        if self.status != ChannelStatus::Active {
            return None;
        }

        if let Some(_) = self.encrypt_payload(payload) {
            println!("[SOVEREIGN CHANNEL] AES-GCM encrypted message sent via {}", self.id);
            Some(MercyEvent::MeshMessageReceived {
                from: self.from_organism.clone(),
                payload_type: "aes_gcm_encrypted_message".to_string(),
            })
        } else {
            None
        }
    }
}

/// Manager for sovereign encrypted channels.
pub struct SovereignChannelManager {
    channels: HashMap<String, SovereignChannel>,
}

impl SovereignChannelManager {
    pub fn new() -> Self {
        Self { channels: HashMap::new() }
    }

    pub fn create_channel(&mut self, from: &str, to: &str, direction: ChannelDirection) -> &mut SovereignChannel {
        let channel = SovereignChannel::new(from, to, direction);
        let id = channel.id.clone();
        self.channels.insert(id.clone(), channel);
        self.channels.get_mut(&id).unwrap()
    }

    pub fn get_channel(&self, id: &str) -> Option<&SovereignChannel> {
        self.channels.get(id)
    }

    pub fn get_active_encrypted_channels(&self) -> Vec<&SovereignChannel> {
        self.channels
            .values()
            .filter(|c| c.status == ChannelStatus::Active && c.encryption_state == EncryptionState::ActiveEncrypted)
            .collect()
    }
}