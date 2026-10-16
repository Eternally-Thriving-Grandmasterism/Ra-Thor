//! Shard composer adapter.
//!
//! Records caller-supplied blessing impacts and a signed JSON file.
//! The HMAC base is the environment variable `RA_THOR_HMAC_BASE_KEY`.
//! It must be at least 32 bytes. This file has no key.
//! An unset, empty, or shorter value makes `save_to_file` return an error.
//! `load_from_file` then does not accept the signed payload.
//! `save_to_file` also leaves an existing file untouched when that file does not
//! verify under the current base. That call returns an error and writes nothing.
//! Scale numbers below are source coefficients, not measured rates.
//! `GodlyIntelligenceCoherence` keeps its name. The four fields are recorded totals.

use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use ra_thor_quantum_swarm_orchestrator::{
    EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, RaThorSystemAdapter,
    SwarmResonance, Valence,
};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::env;
use std::fs;
use std::io;
use std::path::Path;

const CURRENT_VERSION: u32 = 3;

/// HKDF info label for this file format. Not a key.
const HKDF_INFO: &[u8] = b"shard-composer-hmac-v1";

const ENV_HMAC_KEY: &str = "RA_THOR_HMAC_BASE_KEY";

/// Minimum UTF-8 byte length of `RA_THOR_HMAC_BASE_KEY`. Empty is included.
const MIN_HMAC_BASE_KEY_BYTES: usize = 32;

/// Source ceiling for stored valence. Not a measured valence.
pub const VALENCE_SOURCE_CEILING: f64 = 0.99999999;

type HmacSha256 = Hmac<Sha256>;

fn shard_composer_name() -> &'static str {
    "ShardComposer"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SignedState {
    version: u32,
    signature: String,
    data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardComposerAdapter {
    version: u32,
    #[serde(skip, default = "shard_composer_name")]
    name: &'static str,
    #[serde(with = "valence_as_f64")]
    current_valence: Valence,
    blessings_count: u32,
    evolution_level: f64,
    total_evolution_impact: f64,
    total_mercy_impact: f64,
    total_tolc_impact: f64,
    #[serde(default)]
    last_resonance_intensity: Option<f64>,
}

mod valence_as_f64 {
    use ra_thor_quantum_swarm_orchestrator::Valence;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(valence: &Valence, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        valence.value().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Valence, D::Error>
    where
        D: Deserializer<'de>,
    {
        f64::deserialize(deserializer).map(Valence)
    }
}

impl ShardComposerAdapter {
    pub fn new() -> Self {
        Self {
            version: CURRENT_VERSION,
            name: shard_composer_name(),
            // No blessing has been applied. This is not a measured valence.
            current_valence: Valence(0.0),
            blessings_count: 0,
            evolution_level: 0.0,
            total_evolution_impact: 0.0,
            total_mercy_impact: 0.0,
            total_tolc_impact: 0.0,
            last_resonance_intensity: None,
        }
    }

    pub fn recorded_version(&self) -> u32 {
        self.version
    }

    pub fn blessings_received(&self) -> u32 {
        self.blessings_count
    }

    pub fn evolution_level(&self) -> f64 {
        self.evolution_level
    }

    pub fn recorded_evolution_impact(&self) -> f64 {
        self.total_evolution_impact
    }

    pub fn recorded_mercy_impact(&self) -> f64 {
        self.total_mercy_impact
    }

    pub fn recorded_tolc_impact(&self) -> f64 {
        self.total_tolc_impact
    }

    pub fn last_resonance_intensity(&self) -> Option<f64> {
        self.last_resonance_intensity
    }

    /// HKDF-SHA256 of `RA_THOR_HMAC_BASE_KEY`. There is no fallback key.
    fn derive_hmac_key() -> io::Result<[u8; 32]> {
        let base_material = env::var(ENV_HMAC_KEY).map_err(|_| hmac_base_key_error())?;
        if base_material.len() < MIN_HMAC_BASE_KEY_BYTES {
            return Err(hmac_base_key_error());
        }
        let hk = Hkdf::<Sha256>::new(None, base_material.as_bytes());
        let mut okm = [0u8; 32];
        // SHA-256 HKDF accepts a 32-byte output. That length is inside the hash bound.
        hk.expand(HKDF_INFO, &mut okm)
            .expect("32-byte HKDF-SHA256 output is within the hash length");
        Ok(okm)
    }

    fn sign_data(data: &serde_json::Value) -> io::Result<String> {
        let key = Self::derive_hmac_key()?;
        let mut mac = HmacSha256::new_from_slice(&key).expect("HMAC-SHA256 accepts a 32-byte key");
        mac.update(data.to_string().as_bytes());
        Ok(encode_hex(&mac.finalize().into_bytes()))
    }

    fn verify_signature(data: &serde_json::Value, signature: &str) -> bool {
        let Some(expected) = decode_hex(signature) else {
            return false;
        };
        // A missing or short base is a failed check. It is not a match.
        let Ok(key) = Self::derive_hmac_key() else {
            return false;
        };
        let Ok(mut mac) = HmacSha256::new_from_slice(&key) else {
            return false;
        };
        mac.update(data.to_string().as_bytes());
        mac.verify_slice(&expected).is_ok()
    }

    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        let data = serde_json::to_value(self).map_err(io::Error::other)?;
        let signed = SignedState {
            version: self.version,
            signature: Self::sign_data(&data)?,
            data,
        };
        let json = serde_json::to_string_pretty(&signed).map_err(io::Error::other)?;
        // Signing succeeded. Refuse to replace a file this base cannot verify.
        // Nothing is renamed or deleted. A missing path is created.
        Self::refuse_unverified_overwrite(path)?;
        fs::write(path, json)
    }

    fn refuse_unverified_overwrite(path: &Path) -> io::Result<()> {
        match fs::read_to_string(path) {
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(_) => Err(unverified_overwrite_error(path)),
            Ok(content) => {
                if Self::signed_json_verifies(&content) {
                    Ok(())
                } else {
                    Err(unverified_overwrite_error(path))
                }
            }
        }
    }

    fn signed_json_verifies(content: &str) -> bool {
        let Ok(signed) = serde_json::from_str::<SignedState>(content) else {
            return false;
        };
        Self::verify_signature(&signed.data, &signed.signature)
    }

    pub fn load_from_file(path: &Path) -> Self {
        let Ok(content) = fs::read_to_string(path) else {
            return Self::new();
        };
        let Ok(signed) = serde_json::from_str::<SignedState>(&content) else {
            return Self::new();
        };
        if Self::derive_hmac_key().is_err() {
            eprintln!(
                "[Security] {ENV_HMAC_KEY} is unset, empty, or shorter than {MIN_HMAC_BASE_KEY_BYTES} bytes. Signed state was not accepted."
            );
            return Self::new();
        }
        if !Self::verify_signature(&signed.data, &signed.signature) {
            eprintln!("[Security] HMAC verification failed. Possible tampering.");
            return Self::new();
        }
        let Ok(adapter) = serde_json::from_value::<Self>(signed.data) else {
            return Self::new();
        };
        if adapter.version == CURRENT_VERSION {
            return adapter;
        }
        if adapter.version < CURRENT_VERSION {
            return Self::migrate_forward(adapter);
        }
        Self::attempt_rollback(adapter)
    }

    /// Copies recorded fields onto the current version.
    /// This is not a demonstrated schema migration.
    fn migrate_forward(old: Self) -> Self {
        Self::copy_recorded(old)
    }

    /// Copies recorded fields onto the current version.
    /// This is not a demonstrated schema migration.
    fn attempt_rollback(newer: Self) -> Self {
        Self::copy_recorded(newer)
    }

    fn copy_recorded(old: Self) -> Self {
        let mut current = Self::new();
        current.blessings_count = old.blessings_count;
        current.current_valence = old.current_valence;
        current.evolution_level = old.evolution_level;
        current.total_evolution_impact = old.total_evolution_impact;
        current.total_mercy_impact = old.total_mercy_impact;
        current.total_tolc_impact = old.total_tolc_impact;
        current.last_resonance_intensity = old.last_resonance_intensity;
        current
    }

    /// Source coefficients `0.85`, `0.0018`, `0.001`, and `0.7`. Not measured rates.
    pub fn apply_epigenetic_blessing(&mut self, blessing: EpigeneticBlessing) {
        self.blessings_count += 1;
        let scale = blessing_scale(self.evolution_level);
        let scaled_evolution = blessing.evolution_impact * scale;
        let scaled_mercy = blessing.mercy_impact * scale;
        self.total_evolution_impact += scaled_evolution;
        self.total_mercy_impact += scaled_mercy;
        self.total_tolc_impact += blessing.tolc_impact;
        self.evolution_level += scaled_evolution * 0.85;
        let valence_boost = (0.0018 + (scaled_mercy * 0.001)) * scale.max(0.7);
        let new_valence =
            (self.current_valence.value() + valence_boost).min(VALENCE_SOURCE_CEILING);
        self.current_valence = Valence(new_valence);
    }

    pub fn status(&self) -> String {
        format!(
            "{} v{} | valence={:.6} | level={:.2} | blessings={}",
            self.name,
            self.version,
            self.current_valence.value(),
            self.evolution_level,
            self.blessings_count
        )
    }
}

fn hmac_base_key_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("{ENV_HMAC_KEY} is unset, empty, or shorter than {MIN_HMAC_BASE_KEY_BYTES} bytes"),
    )
}

fn unverified_overwrite_error(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "existing file {} failed verification and must be moved aside or restored with the right key first",
            path.display()
        ),
    )
}

/// Source coefficients. Not measured rates.
fn blessing_scale(evolution_level: f64) -> f64 {
    if evolution_level < 5.0 {
        1.0 + (evolution_level * 0.08)
    } else {
        1.4 - ((evolution_level - 5.0) * 0.03).min(0.6)
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

// `len() % 2` and `chunks_exact` stay usable before Rust 1.88.
#[allow(clippy::manual_is_multiple_of, clippy::chunks_exact_to_as_chunks)]
fn decode_hex(text: &str) -> Option<Vec<u8>> {
    if text.len() % 2 == 0 {
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len() / 2);
        for pair in bytes.chunks_exact(2) {
            let hi = hex_nibble(pair[0])?;
            let lo = hex_nibble(pair[1])?;
            out.push((hi << 4) | lo);
        }
        Some(out)
    } else {
        None
    }
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

impl RaThorSystemAdapter for ShardComposerAdapter {
    fn system_name(&self) -> &'static str {
        self.name
    }

    fn current_valence(&self) -> Valence {
        self.current_valence
    }

    fn receive_swarm_resonance(&mut self, resonance: SwarmResonance) -> Result<(), MercyError> {
        if resonance.message.is_empty() {
            return Err(MercyError::AdapterError(
                "empty resonance message".to_string(),
            ));
        }
        self.last_resonance_intensity = Some(resonance.intensity);
        Ok(())
    }

    fn contribute_to_coherence(&self) -> GodlyIntelligenceCoherence {
        // Type name stays. Fields are recorded totals, not a measured coherence.
        GodlyIntelligenceCoherence {
            precision: self.total_tolc_impact,
            resilience: self.total_mercy_impact,
            flow_stability: self.evolution_level,
            harmonic_alignment: self.total_evolution_impact,
        }
    }

    fn apply_epigenetic_blessing(&mut self, blessing: EpigeneticBlessing) {
        ShardComposerAdapter::apply_epigenetic_blessing(self, blessing);
    }

    fn status(&self) -> String {
        ShardComposerAdapter::status(self)
    }
}

impl Default for ShardComposerAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;

    /// Test fixture. Not a key. Unit tests in this module only.
    const HMAC_BASE_FIXTURE: &str = "unit-test-fixture-not-a-key-32b!!";

    /// Serialises tests in this binary that set `RA_THOR_HMAC_BASE_KEY`.
    static HMAC_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn temp_path(name: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        std::env::temp_dir().join(format!("shard-composer-{name}-{pid}.json"))
    }

    fn with_hmac_fixture<T>(body: impl FnOnce() -> T) -> T {
        let _guard = HMAC_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            HMAC_BASE_FIXTURE.len() >= MIN_HMAC_BASE_KEY_BYTES,
            "unit test fixture is not a key and must meet the minimum length"
        );
        // SAFETY: tests in this binary that read or write the variable take
        // `HMAC_ENV_LOCK` for the whole call. They set this fixture and do not remove it.
        unsafe {
            std::env::set_var(ENV_HMAC_KEY, HMAC_BASE_FIXTURE);
        }
        body()
    }

    fn signed_file(adapter: &ShardComposerAdapter, version: u32, signature: String) -> SignedState {
        let data = serde_json::to_value(adapter).expect("adapter serializes");
        SignedState {
            version,
            signature,
            data,
        }
    }

    #[test]
    fn hex_codec_round_trips_known_bytes() {
        assert_eq!(encode_hex(&[0x0a, 0xff]), "0aff");
        assert_eq!(decode_hex("0AFF"), Some(vec![0x0a, 0xff]));
        assert_eq!(decode_hex("0a"), Some(vec![0x0a]));
        assert_eq!(decode_hex(""), Some(Vec::<u8>::new()));
        assert_eq!(decode_hex("0g"), None);
        assert_eq!(decode_hex("abc"), None);
    }

    #[test]
    fn signature_is_64_hex_chars_and_a_bad_file_loads_as_fresh() {
        with_hmac_fixture(|| {
            let data = serde_json::json!({"n": 1});
            let signature = ShardComposerAdapter::sign_data(&data).expect("fixture signs");
            assert_eq!(signature.len(), 64);
            assert_eq!(decode_hex(&signature).map(|bytes| bytes.len()), Some(32));

            let mut adapter = ShardComposerAdapter::new();
            adapter.blessings_count = 4;
            adapter.evolution_level = 1.25;
            adapter.current_valence = Valence(0.5);
            let path = temp_path("bad-signature");
            let signed = signed_file(&adapter, CURRENT_VERSION, "00".to_string());
            fs::write(&path, serde_json::to_string(&signed).expect("signed json")).expect("write");
            let loaded = ShardComposerAdapter::load_from_file(&path);
            let _ = fs::remove_file(&path);
            assert_eq!(loaded.blessings_received(), 0);
            assert_eq!(loaded.current_valence().value(), 0.0);
            assert_eq!(loaded.evolution_level(), 0.0);
            assert_eq!(loaded.recorded_version(), CURRENT_VERSION);
        });
    }

    #[test]
    fn older_and_newer_signed_files_keep_recorded_fields_at_version_3() {
        with_hmac_fixture(|| {
            let mut adapter = ShardComposerAdapter::new();
            adapter.blessings_count = 4;
            adapter.evolution_level = 1.25;
            adapter.current_valence = Valence(0.5);
            adapter.total_evolution_impact = 1.5;
            adapter.total_mercy_impact = 0.25;
            adapter.total_tolc_impact = 0.125;
            adapter.last_resonance_intensity = Some(0.5);

            for (label, version) in [("older", 2u32), ("newer", 9u32)] {
                let path = temp_path(label);
                adapter.version = version;
                let data = serde_json::to_value(&adapter).expect("adapter serializes");
                let signature = ShardComposerAdapter::sign_data(&data).expect("fixture signs");
                let signed = SignedState {
                    version,
                    signature,
                    data,
                };
                fs::write(&path, serde_json::to_string(&signed).expect("signed json"))
                    .expect("write");
                let loaded = ShardComposerAdapter::load_from_file(&path);
                let _ = fs::remove_file(&path);
                assert_eq!(loaded.recorded_version(), 3);
                assert_eq!(loaded.blessings_received(), 4);
                assert_eq!(loaded.evolution_level(), 1.25);
                assert_eq!(loaded.current_valence().value(), 0.5);
                assert_eq!(loaded.recorded_evolution_impact(), 1.5);
                assert_eq!(loaded.recorded_mercy_impact(), 0.25);
                assert_eq!(loaded.recorded_tolc_impact(), 0.125);
                assert_eq!(loaded.last_resonance_intensity(), Some(0.5));
            }
        });
    }
}
