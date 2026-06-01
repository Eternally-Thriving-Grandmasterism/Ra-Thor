//! adapter.rs
//!
//! Tamper-resistant persistence with HKDF + environment variable support.

use ra_thor_quantum_swarm_orchestrator::{
    adapter::RaThorSystemAdapter,
    types::{EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, SwarmResonance, Valence},
};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::env;
use std::fs;
use std::path::Path;

const CURRENT_VERSION: u32 = 2;

// Default base material (used if env var is not set)
const DEFAULT_BASE_KEY: &[u8] = b"ra-thor-sovereign-self-evolution";
const HKDF_INFO: &[u8] = b"shard-composer-hmac-v1";

const ENV_HMAC_KEY: &str = "RA_THOR_HMAC_BASE_KEY";

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SignedState {
    version: u32,
    signature: String,
    data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardComposerAdapter {
    version: u32,
    name: &'static str,
    current_valence: Valence,
    #[serde(default)]
    blessings_count: u32,
}

impl ShardComposerAdapter {
    pub fn new() -> Self {
        Self {
            version: CURRENT_VERSION,
            name: "ShardComposer",
            current_valence: Valence(0.99999995),
            blessings_count: 0,
        }
    }

    pub fn blessings_received(&self) -> u32 {
        self.blessings_count
    }

    /// Derive HMAC key using HKDF, with optional env var override
    fn derive_hmac_key() -> Vec<u8> {
        let base_material = env::var(ENV_HMAC_KEY)
            .map(|s| s.into_bytes())
            .unwrap_or_else(|_| DEFAULT_BASE_KEY.to_vec());

        let hk = Hkdf::<Sha256>::new(None, &base_material);
        let mut okm = [0u8; 32];
        hk.expand(HKDF_INFO, &mut okm).expect("HKDF expand failed");
        okm.to_vec()
    }

    fn sign_data(data: &serde_json::Value) -> Option<String> {
        let key = Self::derive_hmac_key();
        let mut mac = HmacSha256::new_from_slice(&key).ok()?;
        mac.update(data.to_string().as_bytes());
        let result = mac.finalize();
        Some(hex::encode(result.into_bytes()))
    }

    fn verify_signature(data: &serde_json::Value, signature: &str) -> bool {
        let key = Self::derive_hmac_key();
        if let Ok(mut mac) = HmacSha256::new_from_slice(&key) {
            mac.update(data.to_string().as_bytes());
            if let Ok(expected) = hex::decode(signature) {
                return mac.verify_slice(&expected).is_ok();
            }
        }
        false
    }

    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let data = serde_json::to_value(self)?;
        let signature = Self::sign_data(&data).unwrap_or_default();

        let signed = SignedState {
            version: self.version,
            signature,
            data,
        };

        let json = serde_json::to_string_pretty(&signed)?;
        fs::write(path, json)
    }

    pub fn load_from_file(path: &Path) -> Self {
        if let Ok(content) = fs::read_to_string(path) {
            if let Ok(signed) = serde_json::from_str::<SignedState>(&content) {
                if !Self::verify_signature(&signed.data, &signed.signature) {
                    eprintln!("[Security] HMAC verification failed. Possible tampering.");
                    return Self::new();
                }

                if let Ok(mut adapter) = serde_json::from_value::<ShardComposerAdapter>(signed.data) {
                    if adapter.version == CURRENT_VERSION {
                        return adapter;
                    }
                    if adapter.version < CURRENT_VERSION {
                        return Self::migrate_forward(adapter);
                    }
                    if adapter.version > CURRENT_VERSION {
                        return Self::attempt_rollback(adapter);
                    }
                }
            }
        }
        Self::new()
    }

    fn migrate_forward(old: ShardComposerAdapter) -> Self {
        let mut new = Self::new();
        new.blessings_count = old.blessings_count;
        new.current_valence = old.current_valence;
        new
    }

    fn attempt_rollback(newer: ShardComposerAdapter) -> Self {
        let mut adapter = Self::new();
        adapter.blessings_count = newer.blessings_count;
        adapter.current_valence = newer.current_valence;
        adapter
    }
}

impl RaThorSystemAdapter for ShardComposerAdapter {
    fn system_name(&self) -> &'static str {
        self.name
    }

    fn current_valence(&self) -> Valence {
        self.current_valence
    }

    fn receive_swarm_resonance(
        &mut self,
        resonance: SwarmResonance,
    ) -> Result<(), MercyError> {
        println!("[ShardComposer] Received resonance: {}", resonance.message);
        Ok(())
    }

    fn contribute_to_coherence(&self) -> GodlyIntelligenceCoherence {
        GodlyIntelligenceCoherence {
            precision: 0.92,
            resilience: 0.88,
            flow_stability: 0.90,
            harmonic_alignment: 0.85,
        }
    }

    fn apply_epigenetic_blessing(&mut self, blessing: EpigeneticBlessing) {
        self.blessings_count += 1;
        let new_valence = (self.current_valence.value() + blessing.strength * 0.001).min(0.99999999);
        self.current_valence = Valence(new_valence);
    }

    fn status(&self) -> String {
        format!(
            "{} v{}: valence={:.6}, blessings={}",
            self.name,
            self.version,
            self.current_valence.value(),
            self.blessings_count
        )
    }
}
