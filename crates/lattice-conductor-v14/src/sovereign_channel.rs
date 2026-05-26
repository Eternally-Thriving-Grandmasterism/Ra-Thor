//! Sovereign Channel Encryption — Thunder Lattice v14.0.7+
//! Production-grade prototype for mercy-gated, sovereign end-to-end encrypted channels.

use crate::distributed_mercy_mesh::MercyEvent;
use std::collections::HashMap;

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

/// Represents the encryption state of a sovereign channel.
#[derive(Debug, Clone, PartialEq)]
pub enum EncryptionState {
    Unencrypted,
    KeyEstablished,
    ActiveEncrypted,
}

/// A sovereign, mercy-gated encrypted communication channel between two organisms.
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
    // In production this would be a proper key handle / key material
    channel_key: Option<Vec<u8>>,
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

    /// Establish a shared secret (prototype for ECDH / pre-shared key).
    /// In real implementation this would perform proper key agreement with mercy validation.
    pub fn establish_encryption(&mut self, shared_secret: Vec<u8>) {
        self.channel_key = Some(shared_secret);
        self.encryption_state = EncryptionState::KeyEstablished;
        println!("[SOVEREIGN CHANNEL] Encryption key established for {}", self.id);
    }

    pub fn activate(&mut self) {
        if self.encryption_state == EncryptionState::KeyEstablished {
            self.encryption_state = EncryptionState::ActiveEncrypted;
        }
        self.status = ChannelStatus::Active;
        println!("[SOVEREIGN CHANNEL] Channel {} activated (encrypted={})", 
            self.id, self.encryption_state == EncryptionState::ActiveEncrypted);
    }

    pub fn close(&mut self) {
        self.status = ChannelStatus::Closed;
        self.encryption_state = EncryptionState::Unencrypted;
        self.channel_key = None;
    }

    /// Encrypt a payload using the channel key (prototype implementation).
    /// Real version would use AES-GCM / ChaCha20-Poly1305 with proper nonce + mercy signature.
    pub fn encrypt_payload(&self, plaintext: &[u8]) -> Option<Vec<u8>> {
        if self.encryption_state != EncryptionState::ActiveEncrypted {
            return None;
        }
        let key = self.channel_key.as_ref()?;

        // Production-grade note: Replace with real AEAD cipher (AES-GCM recommended)
        // This is a simple XOR + length prefix prototype for sovereignty demonstration
        let mut ciphertext = Vec::with_capacity(plaintext.len() + 4);
        ciphertext.extend_from_slice(&(plaintext.len() as u32).to_be_bytes());

        for (i, byte) in plaintext.iter().enumerate() {
            let key_byte = key[i % key.len()];
            ciphertext.push(byte ^ key_byte);
        }

        Some(ciphertext)
    }

    /// Decrypt a payload using the channel key.
    pub fn decrypt_payload(&self, ciphertext: &[u8]) -> Option<Vec<u8>> {
        if self.encryption_state != EncryptionState::ActiveEncrypted || ciphertext.len() < 4 {
            return None;
        }
        let key = self.channel_key.as_ref()?;

        let len = u32::from_be_bytes([ciphertext[0], ciphertext[1], ciphertext[2], ciphertext[3]]) as usize;
        if ciphertext.len() != len + 4 {
            return None;
        }

        let mut plaintext = Vec::with_capacity(len);
        for (i, byte) in ciphertext[4..].iter().enumerate() {
            let key_byte = key[i % key.len()];
            plaintext.push(byte ^ key_byte);
        }

        Some(plaintext)
    }

    /// Send an encrypted message through the channel.
    pub fn send_encrypted_message(&self, payload: &[u8]) -> Option<MercyEvent> {
        if self.status != ChannelStatus::Active {
            return None;
        }

        if let Some(ciphertext) = self.encrypt_payload(payload) {
            println!("[SOVEREIGN CHANNEL] Encrypted message sent via {} ({} bytes)", self.id, ciphertext.len());
            Some(MercyEvent::MeshMessageReceived {
                from: self.from_organism.clone(),
                payload_type: "encrypted_sovereign_message".to_string(),
            })
        } else {
            None
        }
    }
}

/// Manager for multiple sovereign encrypted channels.
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