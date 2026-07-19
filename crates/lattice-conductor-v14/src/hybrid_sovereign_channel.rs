//! Hybrid Sovereign Channel — Classical + Post-Quantum KEM surface (v14.8.2)
//! Production-ready structure; real Kyber/ML-KEM can be wired behind a feature later.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct HybridKeyMaterial {
    pub classical_shared_secret: Option<[u8; 32]>,
    pub pq_shared_secret: Option<Vec<u8>>,
    pub combined_key: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChannelStatus {
    Pending,
    KeyEstablishment,
    Active,
    Closed,
}

#[derive(Debug, Clone)]
pub struct HybridSovereignChannel {
    pub id: String,
    pub from_organism: String,
    pub to_organism: String,
    pub status: ChannelStatus,
    pub hybrid_key: Option<HybridKeyMaterial>,
    pub use_hybrid: bool,
}

impl HybridSovereignChannel {
    pub fn new(from: &str, to: &str) -> Self {
        Self {
            id: format!("hybrid_channel_{}_{}", from, to),
            from_organism: from.into(),
            to_organism: to.into(),
            status: ChannelStatus::Pending,
            hybrid_key: None,
            use_hybrid: true,
        }
    }

    pub fn establish_classical_key(&mut self, key: [u8; 32]) {
        if self.hybrid_key.is_none() {
            self.hybrid_key = Some(HybridKeyMaterial {
                classical_shared_secret: Some(key),
                pq_shared_secret: None,
                combined_key: Some(key),
            });
        } else if let Some(ref mut km) = self.hybrid_key {
            km.classical_shared_secret = Some(key);
            km.combined_key = Some(key);
        }
        self.status = ChannelStatus::KeyEstablishment;
    }

    pub fn establish_post_quantum_secret(&mut self, pq_shared_secret: Vec<u8>) {
        if self.hybrid_key.is_none() {
            self.hybrid_key = Some(HybridKeyMaterial {
                classical_shared_secret: None,
                pq_shared_secret: Some(pq_shared_secret),
                combined_key: None,
            });
        } else if let Some(ref mut km) = self.hybrid_key {
            km.pq_shared_secret = Some(pq_shared_secret);
        }
        self.status = ChannelStatus::KeyEstablishment;
    }

    pub fn finalize_hybrid_key(&mut self) {
        if let Some(ref mut km) = self.hybrid_key {
            if let (Some(classical), Some(pq)) = (&km.classical_shared_secret, &km.pq_shared_secret)
            {
                let mut combined = [0u8; 32];
                for i in 0..32 {
                    combined[i] = classical[i] ^ pq.get(i).copied().unwrap_or(0);
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

    pub fn get_aes_gcm_key(&self) -> Option<[u8; 32]> {
        self.hybrid_key.as_ref().and_then(|km| km.combined_key)
    }
}

pub struct HybridSovereignChannelManager {
    channels: HashMap<String, HybridSovereignChannel>,
}

impl HybridSovereignChannelManager {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    pub fn create_channel(&mut self, from: &str, to: &str) -> &mut HybridSovereignChannel {
        let channel = HybridSovereignChannel::new(from, to);
        let id = channel.id.clone();
        self.channels.insert(id.clone(), channel);
        self.channels.get_mut(&id).expect("just inserted")
    }

    pub fn get_active_channels(&self) -> Vec<&HybridSovereignChannel> {
        self.channels.values().filter(|c| c.is_active()).collect()
    }
}

impl Default for HybridSovereignChannelManager {
    fn default() -> Self {
        Self::new()
    }
}
