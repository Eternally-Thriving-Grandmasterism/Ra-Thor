//! Exact values for the compiled council wire.
//! Threshold figures are source literals.

use std::fs;
use std::path::Path;

use quantum_gravity_harmony_council::{
    mean_bond_valence, QuantumGravityHarmonyCouncil, REASONING_THRESHOLD_SOURCE_LITERAL,
};

const UNCOMPILED_TEXT: &str = include_str!("../src/uncompiled_council.rs");
const UNCOMPILED_BODY_LEN: usize = 2070;
const UNCOMPILED_BODY_CHECKSUM: u64 = 44192849677582746;

fn preserved_body() -> &'static str {
    let marker = "#![rustfmt::skip]\n";
    let (head, body) = UNCOMPILED_TEXT.split_once(marker).unwrap();
    assert!(head.starts_with(
        "//! Design-only (STUB-AUDIT-1 REASONING-COUNCIL-WIRES-1 council note): not compiled;",
    ),);
    assert_eq!(UNCOMPILED_TEXT.lines().nth(1), Some("#![rustfmt::skip]"));
    body
}

fn body_checksum(body: &str) -> u64 {
    body.bytes().fold(0u64, |acc, byte| {
        acc.wrapping_mul(16777619).wrapping_add(u64::from(byte))
    })
}

#[test]
fn uncompiled_body_matches_the_preserved_bytes() {
    let body = preserved_body();
    assert_eq!(body.len(), UNCOMPILED_BODY_LEN);
    assert_eq!(body_checksum(body), UNCOMPILED_BODY_CHECKSUM);
    let lib = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs")).unwrap();
    assert_eq!(lib.matches("mod uncompiled_council").count(), 0);
}

#[test]
fn record_stores_the_historical_title() {
    let council = QuantumGravityHarmonyCouncil::new();
    assert_eq!(council.id, 22);
    assert_eq!(council.name, "Quantum Gravity Harmony Council");
    assert_eq!(QuantumGravityHarmonyCouncil::default().id, 22);
    assert_eq!(QuantumGravityHarmonyCouncil::default().name, council.name);
    assert_eq!(REASONING_THRESHOLD_SOURCE_LITERAL, 0.9999999);
}

#[test]
fn reasoning_error_and_foresight_follow_the_inputs() {
    let low_one = QuantumGravityHarmonyCouncil::reasoning_from_inputs(0.5, 1).unwrap_err();
    let low_two = QuantumGravityHarmonyCouncil::reasoning_from_inputs(0.5, 2).unwrap_err();
    assert_eq!(low_one, low_two);
    assert_eq!(low_one, "Mercy Gate violation: valence below threshold");
    for years in [1_u64, 2, 1_000_000] {
        let got = QuantumGravityHarmonyCouncil::reasoning_from_inputs(
            REASONING_THRESHOLD_SOURCE_LITERAL,
            years,
        )
        .unwrap();
        let expect = (0.9999999_f64 + (years as f64).ln() * 0.0000001).min(0.99999999);
        assert_eq!(got, expect);
    }
    let year_one =
        QuantumGravityHarmonyCouncil::reasoning_from_inputs(REASONING_THRESHOLD_SOURCE_LITERAL, 1)
            .unwrap();
    let year_two =
        QuantumGravityHarmonyCouncil::reasoning_from_inputs(REASONING_THRESHOLD_SOURCE_LITERAL, 2)
            .unwrap();
    assert_ne!(year_one, year_two);
}

#[test]
fn bond_mean_is_the_mean_of_the_supplied_valences() {
    let err = mean_bond_valence(&[]).unwrap_err();
    assert_eq!(err, "no bonds supplied");
    let rejected = mean_bond_valence(&[("a", "b", 0.5)]).unwrap_err();
    assert_eq!(
        rejected,
        "TOLC 8 Sovereignty Gate violation in philotic fusion",
    );
    let bonds = [("a", "b", 1.0), ("c", "d", 1.5), ("e", "f", 2.0)];
    let got = mean_bond_valence(&bonds).unwrap();
    let expect = (1.0 + 1.5 + 2.0) / 3.0;
    assert_eq!(got, expect);
}

#[test]
fn tolc8_mercy_check_follows_the_source_literal_threshold() {
    let council = QuantumGravityHarmonyCouncil::new();
    let threshold = council.valence_threshold;
    assert_eq!(threshold, 0.9999999);
    let below = f64::from_bits(threshold.to_bits() - 1);
    let above = f64::from_bits(threshold.to_bits() + 1);
    assert!(below < threshold);
    assert!(above > threshold);
    assert_eq!(council.tolc8_mercy_check(below), below >= threshold);
    assert_eq!(council.tolc8_mercy_check(threshold), above > below);
    assert_eq!(council.tolc8_mercy_check(above), above >= threshold);
}
