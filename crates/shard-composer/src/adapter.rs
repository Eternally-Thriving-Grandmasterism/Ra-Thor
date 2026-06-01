//! adapter.rs
//!
//! RaThorSystemAdapter with versioning, migration, HMAC, HKDF, and meaningful blessing application.

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
    blessings_count: u32,

    // Accumulated impacts from blessings
    total_evolution_impact: f64,
    total_mercy_impact: f64,
    total_tolc_impact: f64,
}

impl ShardComposerAdapter {
    pub fn new() -> Self {
        Self {
            version: CURRENT_VERSION,
            name: "ShardComposer",
            current_valence: Valence(0.99999995),
            blessings_count: 0,
            total_evolution_impact: 0.0,
            total_mercy_impact: 0.0,
            total_tolc_impact: 0.0,
        }
    }

    pub fn blessings_received(&self) -> u32 {
        self.blessings_count
    }

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
        new.total_evolution_impact = old.total_evolution_impact;
        new.total_mercy_impact = old.total_mercy_impact;
        new.total_tolc_impact = old.total_tolc_impact;
        new
    }

    fn attempt_rollback(newer: ShardComposerAdapter) -> Self {
        let mut adapter = Self::new();
        adapter.blessings_count = newer.blessings_count;
        adapter.current_valence = newer.current_valence;
        adapter.total_evolution_impact = newer.total_evolution_impact;
        adapter.total_mercy_impact = newer.total_mercy_impact;
        adapter.total_tolc_impact = newer.total_tolc_impact;
        adapter
    }

    /// Enhanced application that respects multi-dimensional impacts (evolved from v13)
    pub fn apply_epigenetic_blessing(&mut self, blessing: EpigeneticBlessing) {
        self.blessings_count += 1;

        self.total_evolution_impact += blessing.evolution_impact;
        self.total_mercy_impact += blessing.mercy_impact;
        self.total_tolc_impact += blessing.tolc_impact;

        // Modulate valence improvement based on mercy_impact
        let valence_boost = 0.0008 + (blessing.mercy_impact * 0.0003);
        let new_valence = (self.current_valence.value() + valence_boost).min(0.99999999);
        self.current_valence = Valence(new_valence);

        println!(
            "[ShardComposer] Applied {} | strength={:.2} | evo={:.3} mercy={:.3} tolc={:.3} | total_blessings={}",
            blessing.blessing_type,
            blessing.strength,
            blessing.evolution_impact,
            blessing.mercy_impact,
            blessing.tolc_impact,
            self.blessings_count
        );
    }

    pub fn status(&self) -> String {
        format!(
            "{} v{}: valence={:.6}, blessings={}, evo_impact={:.2}, mercy_impact={:.2}",
            self.name,
            self.version,
            self.current_valence.value(),
            self.blessings_count,
            self.total_evolution_impact,
            self.total_mercy_impact
        )
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
        self.apply_epigenetic_blessing(blessing);
    }

    fn status(&self) -> String {
        self.status()
    }
}
