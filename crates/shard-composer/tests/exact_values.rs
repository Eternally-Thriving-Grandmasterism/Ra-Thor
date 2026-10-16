//! Exact values for the compiled shard composer.
//! Scale numbers are the source coefficients in `adapter`.
//! `VALENCE_SOURCE_CEILING` is a source ceiling, not a measured valence.
//! `Valence::MIN` below is the orchestrator source literal `0.9999999`.

use shard_composer::quantum_swarm::{
    EpigeneticBlessing, MercyError, RaThorSystemAdapter, SwarmResonance, Valence,
};
use shard_composer::{ShardComposerAdapter, VALENCE_SOURCE_CEILING};
use std::fs;
use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    let pid = std::process::id();
    std::env::temp_dir().join(format!("shard-composer-it-{name}-{pid}.json"))
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
fn reexports_are_the_standalone_packages_that_build() {
    assert_eq!(shard_composer::patsagi::VERSION, "14.15.11");
    assert_eq!(shard_composer::quantum_swarm::Valence::MIN, 0.9999999);
}
