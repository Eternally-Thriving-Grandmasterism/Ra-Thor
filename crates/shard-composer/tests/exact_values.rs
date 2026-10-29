//! Exact values for the compiled shard composer.
//! Scale numbers are the source coefficients in `adapter`.
//! `VALENCE_SOURCE_CEILING` is a source ceiling, not a measured valence.
//! `Valence::MIN` below is the orchestrator source literal `0.9999999`.

use shard_composer::quantum_swarm::{
    EpigeneticBlessing, MercyError, RaThorSystemAdapter, SwarmResonance, Valence,
};
use shard_composer::{ShardComposerAdapter, VALENCE_SOURCE_CEILING};
use std::fs;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

/// Serialises tests that set or remove `RA_THOR_HMAC_BASE_KEY`.
/// The variable is process-global.
static HMAC_ENV_LOCK: Mutex<()> = Mutex::new(());

/// Test fixture. Not a key. Exactly 32 bytes, the documented minimum.
const HMAC_FIXTURE_32: &str = "fixture-hmac-base-32-bytes!!!!!!";
/// Test fixture. Not a key. Different bytes from `HMAC_FIXTURE_32`.
const HMAC_FIXTURE_B: &str = "fixture-hmac-base-other-value!!!!";
/// Test fixture. Not a key. One byte under the minimum.
const HMAC_FIXTURE_SHORT: &str = "fixture-hmac-base-31-bytes!!!!!";

fn temp_path(name: &str) -> PathBuf {
    let pid = std::process::id();
    std::env::temp_dir().join(format!("shard-composer-it-{name}-{pid}.json"))
}

fn lock_hmac_env() -> MutexGuard<'static, ()> {
    HMAC_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn set_hmac_fixture(value: &str) {
    // SAFETY: callers hold `HMAC_ENV_LOCK` for the whole test. Every test in
    // this file that reads or writes `RA_THOR_HMAC_BASE_KEY` takes that lock first.
    unsafe {
        std::env::set_var("RA_THOR_HMAC_BASE_KEY", value);
    }
}

fn unset_hmac_fixture() {
    // SAFETY: callers hold `HMAC_ENV_LOCK` for the whole test.
    unsafe {
        std::env::remove_var("RA_THOR_HMAC_BASE_KEY");
    }
}

fn assert_save_error_names_the_env_var(label: &str) {
    let adapter = ShardComposerAdapter::new();
    let path = temp_path(label);
    let _ = fs::remove_file(&path);
    let err = adapter.save_to_file(&path).expect_err(label);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    let message = err.to_string();
    assert!(
        message.contains("RA_THOR_HMAC_BASE_KEY"),
        "{label}: {message}"
    );
    assert!(
        message.contains("32"),
        "{label}: message names the minimum: {message}"
    );
    assert!(!path.exists(), "{label} must not write a file");
}

fn assert_fresh(loaded: &ShardComposerAdapter) {
    assert_eq!(loaded.blessings_received(), 0);
    assert_eq!(loaded.current_valence(), Valence(0.0));
    assert_eq!(loaded.evolution_level(), 0.0);
    assert_eq!(loaded.recorded_evolution_impact(), 0.0);
    assert_eq!(loaded.recorded_mercy_impact(), 0.0);
    assert_eq!(loaded.recorded_tolc_impact(), 0.0);
    assert_eq!(loaded.recorded_version(), 3);
    assert_eq!(loaded.last_resonance_intensity(), None);
}

fn blessing(evolution: f64, mercy: f64, tolc: f64) -> EpigeneticBlessing {
    EpigeneticBlessing::with_impacts("caller-input", 1.0, "ShardComposer", evolution, mercy, tolc)
}

#[test]
fn fresh_adapter_records_zeros_and_the_source_ceiling_literal() {
    let adapter = ShardComposerAdapter::new();
    assert_eq!(adapter.system_name(), "ShardComposer");
    assert_eq!(adapter.recorded_version(), 3);
    assert_eq!(adapter.blessings_received(), 0);
    assert_eq!(adapter.evolution_level(), 0.0);
    assert_eq!(adapter.recorded_evolution_impact(), 0.0);
    assert_eq!(adapter.recorded_mercy_impact(), 0.0);
    assert_eq!(adapter.recorded_tolc_impact(), 0.0);
    assert_eq!(adapter.current_valence(), Valence(0.0));
    assert_eq!(adapter.last_resonance_intensity(), None);
    assert_eq!(
        adapter.status(),
        "ShardComposer v3 | valence=0.000000 | level=0.00 | blessings=0"
    );
    assert_eq!(VALENCE_SOURCE_CEILING, 0.99999999);
    assert_eq!(ShardComposerAdapter::default().recorded_version(), 3);
    assert_eq!(
        ShardComposerAdapter::default().current_valence(),
        Valence(0.0)
    );

    let coherence = adapter.contribute_to_coherence();
    assert_eq!(coherence.precision, 0.0);
    assert_eq!(coherence.resilience, 0.0);
    assert_eq!(coherence.flow_stability, 0.0);
    assert_eq!(coherence.harmonic_alignment, 0.0);
}

#[test]
fn early_scale_uses_the_source_coefficients() {
    let mut adapter = ShardComposerAdapter::new();
    adapter.apply_epigenetic_blessing(blessing(1.0, 2.0, 0.25));
    let level1 = 1.0 * 0.85;
    assert_eq!(adapter.blessings_received(), 1);
    assert_eq!(adapter.evolution_level(), level1);
    assert_eq!(adapter.recorded_evolution_impact(), 1.0);
    assert_eq!(adapter.recorded_mercy_impact(), 2.0);
    assert_eq!(adapter.recorded_tolc_impact(), 0.25);
    assert_eq!(adapter.current_valence().value(), 0.0018 + (2.0 * 0.001));

    adapter.apply_epigenetic_blessing(blessing(1.0, 0.0, 0.5));
    let scale2 = 1.0 + (level1 * 0.08);
    let scaled_evolution = 1.0 * scale2;
    assert_eq!(adapter.blessings_received(), 2);
    assert_eq!(adapter.evolution_level(), level1 + scaled_evolution * 0.85);
    assert_eq!(adapter.recorded_evolution_impact(), 1.0 + scaled_evolution);
    assert_eq!(adapter.recorded_mercy_impact(), 2.0);
    assert_eq!(adapter.recorded_tolc_impact(), 0.75);
    assert_eq!(
        adapter.current_valence().value(),
        (0.0018 + (2.0 * 0.001)) + ((0.0018 + (0.0 * 0.001)) * scale2.max(0.7))
    );

    let coherence = adapter.contribute_to_coherence();
    assert_eq!(coherence.precision, 0.75);
    assert_eq!(coherence.resilience, 2.0);
    assert_eq!(coherence.flow_stability, level1 + scaled_evolution * 0.85);
    assert_eq!(coherence.harmonic_alignment, 1.0 + scaled_evolution);
}

#[test]
fn late_scale_uses_the_diminishing_source_coefficients() {
    let mut adapter = ShardComposerAdapter::new();
    adapter.apply_epigenetic_blessing(blessing(8.0, 0.0, 0.0));
    let level1 = 8.0 * 0.85;
    assert_eq!(adapter.evolution_level(), level1);
    assert_eq!(adapter.current_valence().value(), 0.0018);

    adapter.apply_epigenetic_blessing(blessing(2.0, 4.0, 0.5));
    let scale = 1.4 - ((level1 - 5.0) * 0.03).min(0.6);
    let scaled_evolution = 2.0 * scale;
    let scaled_mercy = 4.0 * scale;
    assert_eq!(adapter.blessings_received(), 2);
    assert_eq!(adapter.evolution_level(), level1 + scaled_evolution * 0.85);
    assert_eq!(adapter.recorded_evolution_impact(), 8.0 + scaled_evolution);
    assert_eq!(adapter.recorded_mercy_impact(), scaled_mercy);
    assert_eq!(adapter.recorded_tolc_impact(), 0.5);
    assert_eq!(
        adapter.current_valence().value(),
        (0.0018 + ((0.0018 + (scaled_mercy * 0.001)) * scale.max(0.7))).min(VALENCE_SOURCE_CEILING)
    );
}

#[test]
fn valence_boost_stops_at_the_source_ceiling() {
    let mut adapter = ShardComposerAdapter::new();
    adapter.apply_epigenetic_blessing(blessing(0.0, 10_000.0, 0.0));
    let uncapped = 0.0018 + (10_000.0 * 0.001);
    assert_eq!(adapter.current_valence().value(), VALENCE_SOURCE_CEILING);
    assert_eq!(adapter.current_valence().value(), 0.99999999);
    assert_ne!(adapter.current_valence().value(), uncapped);
    assert_eq!(adapter.recorded_mercy_impact(), 10_000.0);
    assert_eq!(adapter.evolution_level(), 0.0);
}

#[test]
fn resonance_stores_the_caller_intensity() {
    let mut adapter = ShardComposerAdapter::new();
    let empty = adapter.receive_swarm_resonance(SwarmResonance {
        source: "caller".to_string(),
        intensity: 0.5,
        message: String::new(),
    });
    match empty {
        Err(MercyError::AdapterError(text)) => assert_eq!(text, "empty resonance message"),
        Err(other) => panic!("unexpected mercy error: {other}"),
        Ok(()) => panic!("empty message is not stored"),
    }
    assert_eq!(adapter.last_resonance_intensity(), None);

    let stored = adapter.receive_swarm_resonance(SwarmResonance {
        source: "caller".to_string(),
        intensity: 0.25,
        message: "ping".to_string(),
    });
    match stored {
        Ok(()) => {}
        Err(other) => panic!("message is stored: {other}"),
    }
    assert_eq!(adapter.last_resonance_intensity(), Some(0.25));

    let zero = adapter.receive_swarm_resonance(SwarmResonance {
        source: "caller".to_string(),
        intensity: 0.0,
        message: "zero".to_string(),
    });
    match zero {
        Ok(()) => {}
        Err(other) => panic!("zero intensity is stored: {other}"),
    }
    assert_eq!(adapter.last_resonance_intensity(), Some(0.0));
}

#[test]
fn save_and_load_keep_recorded_fields_and_a_tampered_file_is_fresh() {
    let _guard = lock_hmac_env();
    set_hmac_fixture(HMAC_FIXTURE_32);
    let mut adapter = ShardComposerAdapter::new();
    adapter.apply_epigenetic_blessing(blessing(1.0, 2.0, 0.25));
    let stored = adapter.receive_swarm_resonance(SwarmResonance {
        source: "caller".to_string(),
        intensity: 0.25,
        message: "ping".to_string(),
    });
    match stored {
        Ok(()) => {}
        Err(other) => panic!("message is stored: {other}"),
    }

    let path = temp_path("roundtrip");
    adapter.save_to_file(&path).expect("save");
    let loaded = ShardComposerAdapter::load_from_file(&path);
    assert_eq!(loaded.recorded_version(), 3);
    assert_eq!(loaded.blessings_received(), 1);
    assert_eq!(loaded.evolution_level(), adapter.evolution_level());
    assert_eq!(loaded.current_valence(), adapter.current_valence());
    assert_eq!(loaded.recorded_evolution_impact(), 1.0);
    assert_eq!(loaded.recorded_mercy_impact(), 2.0);
    assert_eq!(loaded.recorded_tolc_impact(), 0.25);
    assert_eq!(loaded.last_resonance_intensity(), Some(0.25));
    assert_eq!(loaded.system_name(), "ShardComposer");

    let raw = fs::read_to_string(&path).expect("read");
    let tampered_raw = raw.replacen("\"blessings_count\": 1", "\"blessings_count\": 9", 1);
    assert_ne!(tampered_raw, raw);
    assert_eq!(tampered_raw.matches("\"blessings_count\": 9").count(), 1);
    fs::write(&path, tampered_raw).expect("tamper");
    let tampered = ShardComposerAdapter::load_from_file(&path);
    let _ = fs::remove_file(&path);
    assert_eq!(tampered.blessings_received(), 0);
    assert_eq!(tampered.current_valence(), Valence(0.0));
    assert_eq!(tampered.evolution_level(), 0.0);
    assert_eq!(tampered.recorded_version(), 3);
}

#[test]
fn missing_and_invalid_files_load_as_fresh() {
    let missing = temp_path("missing");
    let _ = fs::remove_file(&missing);
    let loaded = ShardComposerAdapter::load_from_file(&missing);
    assert_eq!(loaded.blessings_received(), 0);
    assert_eq!(loaded.current_valence(), Valence(0.0));
    assert_eq!(loaded.recorded_version(), 3);

    let path = temp_path("invalid");
    fs::write(&path, "not-json").expect("write");
    let invalid = ShardComposerAdapter::load_from_file(&path);
    let _ = fs::remove_file(&path);
    assert_eq!(invalid.blessings_received(), 0);
    assert_eq!(invalid.evolution_level(), 0.0);
    assert_eq!(invalid.current_valence().value(), 0.0);
}

#[test]
fn save_to_file_errors_when_hmac_base_key_is_unset() {
    let _guard = lock_hmac_env();
    unset_hmac_fixture();
    assert_save_error_names_the_env_var("unset");
}

#[test]
fn save_to_file_errors_when_hmac_base_key_is_empty_or_shorter_than_32_bytes() {
    let _guard = lock_hmac_env();
    assert_eq!(HMAC_FIXTURE_SHORT.len(), 31);
    set_hmac_fixture("");
    assert_save_error_names_the_env_var("empty");
    set_hmac_fixture(HMAC_FIXTURE_SHORT);
    assert_save_error_names_the_env_var("short");
}

#[test]
fn load_from_file_rejects_signed_file_when_hmac_base_key_is_unset() {
    let _guard = lock_hmac_env();
    set_hmac_fixture(HMAC_FIXTURE_32);
    let adapter = blessed_adapter();
    let path = temp_path("unset-load");
    adapter.save_to_file(&path).expect("save under the fixture");
    unset_hmac_fixture();
    let loaded = ShardComposerAdapter::load_from_file(&path);
    let raw = fs::read_to_string(&path).expect("signed file remains on disk");
    let _ = fs::remove_file(&path);
    assert!(
        raw.contains("\"blessings_count\": 1"),
        "the signed payload stays in the file"
    );
    assert_fresh(&loaded);
}

#[test]
fn round_trip_works_when_hmac_base_key_is_set() {
    let _guard = lock_hmac_env();
    assert_eq!(HMAC_FIXTURE_32.len(), 32);
    set_hmac_fixture(HMAC_FIXTURE_32);
    let mut adapter = blessed_adapter();
    let stored = adapter.receive_swarm_resonance(SwarmResonance {
        source: "caller".to_string(),
        intensity: 0.25,
        message: "ping".to_string(),
    });
    match stored {
        Ok(()) => {}
        Err(other) => panic!("message is stored: {other}"),
    }
    let path = temp_path("round-trip-set");
    adapter.save_to_file(&path).expect("save");
    let loaded = ShardComposerAdapter::load_from_file(&path);
    let _ = fs::remove_file(&path);
    assert_eq!(loaded.recorded_version(), adapter.recorded_version());
    assert_eq!(loaded.blessings_received(), adapter.blessings_received());
    assert_eq!(loaded.evolution_level(), adapter.evolution_level());
    assert_eq!(loaded.current_valence(), adapter.current_valence());
    assert_eq!(
        loaded.recorded_evolution_impact(),
        adapter.recorded_evolution_impact()
    );
    assert_eq!(
        loaded.recorded_mercy_impact(),
        adapter.recorded_mercy_impact()
    );
    assert_eq!(
        loaded.recorded_tolc_impact(),
        adapter.recorded_tolc_impact()
    );
    assert_eq!(loaded.last_resonance_intensity(), Some(0.25));
    assert_eq!(loaded.system_name(), "ShardComposer");
}

#[test]
fn load_from_file_rejects_file_signed_under_a_different_base() {
    let _guard = lock_hmac_env();
    assert_ne!(HMAC_FIXTURE_32, HMAC_FIXTURE_B);
    assert!(HMAC_FIXTURE_B.len() >= 32);
    set_hmac_fixture(HMAC_FIXTURE_32);
    let adapter = blessed_adapter();
    let path = temp_path("key-mismatch");
    adapter.save_to_file(&path).expect("save under fixture A");
    set_hmac_fixture(HMAC_FIXTURE_B);
    let rejected = ShardComposerAdapter::load_from_file(&path);
    assert_fresh(&rejected);
    set_hmac_fixture(HMAC_FIXTURE_32);
    let accepted = ShardComposerAdapter::load_from_file(&path);
    let _ = fs::remove_file(&path);
    assert_eq!(accepted.blessings_received(), adapter.blessings_received());
    assert_eq!(accepted.evolution_level(), adapter.evolution_level());
    assert_eq!(accepted.current_valence(), adapter.current_valence());
    assert_eq!(
        accepted.recorded_mercy_impact(),
        adapter.recorded_mercy_impact()
    );
}

#[test]
fn save_under_a_different_base_refuses_to_overwrite() {
    let _guard = lock_hmac_env();
    set_hmac_fixture(HMAC_FIXTURE_32);
    let path = temp_path("overwrite-other-base");
    blessed_adapter()
        .save_to_file(&path)
        .expect("save under fixture A");
    let before = fs::read(&path).expect("read signed bytes");
    set_hmac_fixture(HMAC_FIXTURE_B);
    let err = ShardComposerAdapter::new()
        .save_to_file(&path)
        .expect_err("a different base must not overwrite");
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert_overwrite_message(&err);
    assert_eq!(fs::read(&path).expect("file remains"), before);
    let _ = fs::remove_file(&path);
}

#[test]
fn save_after_unset_key_load_rejection_refuses_to_overwrite() {
    let _guard = lock_hmac_env();
    set_hmac_fixture(HMAC_FIXTURE_32);
    let path = temp_path("overwrite-after-unset");
    blessed_adapter()
        .save_to_file(&path)
        .expect("save under fixture A");
    unset_hmac_fixture();
    let loaded = ShardComposerAdapter::load_from_file(&path);
    assert_fresh(&loaded);
    let before = fs::read(&path).expect("read signed bytes");
    set_hmac_fixture(HMAC_FIXTURE_B);
    let err = loaded
        .save_to_file(&path)
        .expect_err("save after a rejected load must not overwrite");
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert_overwrite_message(&err);
    assert_eq!(fs::read(&path).expect("file remains"), before);
    let _ = fs::remove_file(&path);
}

#[test]
fn round_trip_overwrite_under_the_same_base_works() {
    let _guard = lock_hmac_env();
    set_hmac_fixture(HMAC_FIXTURE_32);
    let path = temp_path("overwrite-same-base");
    let adapter = blessed_adapter();
    adapter.save_to_file(&path).expect("first save");
    let mut updated = adapter.clone();
    updated.apply_epigenetic_blessing(blessing(1.0, 0.0, 0.5));
    updated
        .save_to_file(&path)
        .expect("overwrite under the same base");
    let loaded = ShardComposerAdapter::load_from_file(&path);
    let _ = fs::remove_file(&path);
    assert_eq!(loaded.blessings_received(), 2);
    assert_eq!(loaded.blessings_received(), updated.blessings_received());
    assert_eq!(loaded.evolution_level(), updated.evolution_level());
    assert_eq!(loaded.current_valence(), updated.current_valence());
    assert_eq!(
        loaded.recorded_tolc_impact(),
        updated.recorded_tolc_impact()
    );
}

#[test]
fn save_to_file_leaves_no_temp_file_on_success() {
    let _guard = lock_hmac_env();
    set_hmac_fixture(HMAC_FIXTURE_32);
    let path = temp_path("atomic-success");
    let _ = fs::remove_file(&path);
    blessed_adapter()
        .save_to_file(&path)
        .expect("save writes the target");
    assert!(path.is_file(), "target exists after save");
    let parent = path.parent().expect("target parent");
    let target_name = path
        .file_name()
        .expect("target file name")
        .to_string_lossy()
        .into_owned();
    let tmp = path.with_file_name(format!("{target_name}.tmp"));
    assert!(!tmp.exists(), "temp remains at {}", tmp.display());
    let leftovers: Vec<String> = fs::read_dir(parent)
        .expect("read target directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| {
            (name.starts_with(&target_name) && name != &target_name)
                || name.starts_with(&format!(".{target_name}."))
        })
        .collect();
    let _ = fs::remove_file(&path);
    assert!(leftovers.is_empty(), "temp file left behind: {leftovers:?}");
}

#[test]
fn two_saves_in_a_row_leave_no_temp_files() {
    let _guard = lock_hmac_env();
    set_hmac_fixture(HMAC_FIXTURE_32);
    let dir = std::env::temp_dir().join(format!(
        "shard-composer-it-two-saves-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create save directory");
    let path = dir.join("state.json");
    let adapter = blessed_adapter();
    adapter.save_to_file(&path).expect("first save");
    assert_dir_has_only(&dir, "state.json");
    let mut updated = adapter.clone();
    updated.apply_epigenetic_blessing(blessing(1.0, 0.0, 0.5));
    updated.save_to_file(&path).expect("second save");
    assert_dir_has_only(&dir, "state.json");
    let loaded = ShardComposerAdapter::load_from_file(&path);
    let _ = fs::remove_dir_all(&dir);
    assert_eq!(loaded.blessings_received(), 2);
    assert_eq!(loaded.blessings_received(), updated.blessings_received());
    assert_eq!(loaded.evolution_level(), updated.evolution_level());
}

fn assert_dir_has_only(dir: &std::path::Path, target_name: &str) {
    let mut names: Vec<String> = fs::read_dir(dir)
        .expect("read target directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    assert_eq!(names, vec![target_name.to_string()]);
    assert!(!dir.join(format!("{target_name}.tmp")).exists());
}

fn assert_overwrite_message(err: &std::io::Error) {
    let message = err.to_string();
    assert!(message.contains("failed verification"), "{message}");
    assert!(message.contains("moved aside"), "{message}");
    assert!(message.contains("restored with the right key"), "{message}");
}

/// Removed documentation example. Not a key. Must match `REJECTED_DOC_PLACEHOLDER`.
const REJECTED_DOC_PLACEHOLDER: &str = "your-strong-secret-at-least-32-bytes";

#[test]
fn save_and_load_reject_the_removed_doc_placeholder() {
    let _guard = lock_hmac_env();
    assert!(REJECTED_DOC_PLACEHOLDER.len() >= 32);
    set_hmac_fixture(HMAC_FIXTURE_32);
    let path = temp_path("rejected-placeholder");
    blessed_adapter()
        .save_to_file(&path)
        .expect("save under the fixture");
    let before = fs::read(&path).expect("read signed bytes");
    set_hmac_fixture(REJECTED_DOC_PLACEHOLDER);
    let fresh_path = temp_path("rejected-placeholder-new");
    let err = ShardComposerAdapter::new()
        .save_to_file(&fresh_path)
        .expect_err("placeholder must not sign");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    let message = err.to_string();
    assert!(message.contains("RA_THOR_HMAC_BASE_KEY"), "{message}");
    assert!(!fresh_path.exists(), "placeholder must not write a file");
    let loaded = ShardComposerAdapter::load_from_file(&path);
    assert_fresh(&loaded);
    assert_eq!(fs::read(&path).expect("signed file unchanged"), before);
    let _ = fs::remove_file(&path);
}

fn blessed_adapter() -> ShardComposerAdapter {
    let mut adapter = ShardComposerAdapter::new();
    adapter.apply_epigenetic_blessing(blessing(1.0, 2.0, 0.25));
    adapter
}

#[test]
fn reexports_are_the_standalone_packages_that_build() {
    assert_eq!(shard_composer::patsagi::VERSION, "14.15.11");
    assert_eq!(shard_composer::quantum_swarm::Valence::MIN, 0.9999999);
}
