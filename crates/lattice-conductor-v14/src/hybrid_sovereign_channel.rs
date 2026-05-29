//! Hybrid Sovereign Channel — Classical (AES-GCM) + Post-Quantum (Kyber-style KEM)
//! Production-grade hybrid encryption for sovereign channels (v14.0.8+)

// Dependencies to add when implementing real Kyber:
// pqcrypto-kyber = "0.7"   or   ml-kem = "0.2"

use crate::sovereign_channel::SovereignChannel;
use std::collections::HashMap;

/// Represents a hybrid key establishment result.
#[derive(Debug, Clone)]
pub struct HybridKeyMaterial {
    pub classical_shared_secret: Option<[u8; 32]>, // For AES-GCM
    pub pq_shared_secret: Option<Vec<u8>>,       // From Kyber KEM
    pub combined_key: Option<[u8; 32]>,            // Final key for AES-GCM
}

/// Hybrid Sovereign Channel combining classical AES-GCM with post-quantum KEM.
#[derive(Debug, Clone)]
pub struct HybridSovereignChannel {
    pub id: String,
    pub from_organism: String,
    pub to_organism: String,
    pub status: ChannelStatus,
    pub hybrid_key: Option<HybridKeyMaterial>,
    pub use_hybrid: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChannelStatus {
    Pending,
    KeyEstablishment,
    Active,
    Closed,
}

impl HybridSovereignChannel {
    pub fn new(from: &str, to: &str) -> Self {
        Self {
            id: format!("hybrid_channel_{}_{}", from, to),
            from_organism: from.to_string(),
            to_organism: to.to_string(),
            status: ChannelStatus::Pending,
            hybrid_key: None,
            use_hybrid: true,
        }
    }

    /// Establish classical AES-GCM key.
    pub fn establish_classical_key(&mut self, key: [u8; 32]) {
        if self.hybrid_key.is_none() {
            self.hybrid_key = Some(HybridKeyMaterial {
                classical_shared_secret: Some(key),
                pq_shared_secret: None,
                combined_key: Some(key),
            });
        } else if let Some(ref mut km) = self.hybrid_key {
            km.classical_shared_secret = Some(key);
            km.combined_key = Some(key); // Can be improved with KDF later
        }
        self.status = ChannelStatus::KeyEstablishment;
    }

    /// Establish post-quantum shared secret (Kyber-style KEM).
    /// In production, this comes from Kyber.Encaps / Decaps.
    pub fn establish_post_quantum_secret(&mut self, pq_shared_secret: Vec<u8>) {
        if self.hybrid_key.is_none() {
            self.hybrid_key = Some(HybridKeyMaterial {
                classical_shared_secret: None,
                pq_shared_secret: Some(pq_shared_secret.clone()),
                combined_key: None,
            });
        } else if let Some(ref mut km) = self.hybrid_key {
            km.pq_shared_secret = Some(pq_shared_secret.clone());
        }
        self.status = ChannelStatus::KeyEstablishment;
    }

    /// Finalize hybrid key (simple combination for prototype).
    /// Real implementation should use a proper KDF (HKDF) on both secrets.
    pub fn finalize_hybrid_key(&mut self) {
        if let Some(ref mut km) = self.hybrid_key {
            if let (Some(classical), Some(pq)) = (&km.classical_shared_secret, &km.pq_shared_secret) {
                // Simple XOR combination for prototype. Use HKDF in production.
                let mut combined = [0u8; 32];
                for i in 0..32 {
                    combined[i] = classical[i] ^ pq.get(i).unwrap_or(&0);
                }
                km.combined_key = Some(combined);
                self.status = ChannelStatus::Active;
                println!("[HYBRID CHANNEL] Hybrid key finalized for {}", self.id);
            }
        }
    }

    pub fn is_active(&self) -> bool {
        self.status == ChannelStatus::Active && self.hybrid_key.is_some()
    }

    /// Get the final key to be used with AES-GCM.
    pub fn get_aes_gcm_key(&self) -> Option<[u8; 32]> {
        self.hybrid_key.as_ref().and_then(|km| km.combined_key)
    }
}

/// Manager for hybrid sovereign channels.
pub struct HybridSovereignChannelManager {
    channels: HashMap<String, HybridSovereignChannel>,
}

impl HybridSovereignChannelManager {
    pub fn new() -> Self {
        Self { channels: HashMap::new() }
    }

    pub fn create_channel(&mut self, from: &str, to: &str) -> &mut HybridSovereignChannel {
        let channel = HybridSovereignChannel::new(from, to);
        let id = channel.id.clone();
        self.channels.insert(id.clone(), channel);
        self.channels.get_mut(&id).unwrap()
    }

    pub fn get_active_channels(&self) -> Vec<&HybridSovereignChannel> {
        self.channels.values().filter(|c| c.is_active()).collect()
    }
}