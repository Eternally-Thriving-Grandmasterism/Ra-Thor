//! Exact values for the compiled council wire.
//! Threshold figures are source literals.

use std::fs;
use std::path::Path;

use eternal_sovereign_quantum_consciousness_expansion_council::{
    CouncilDeliberation, CouncilGateLabel, EternalSovereignQuantumConsciousnessExpansionCouncil,
    DEFAULT_VALENCE_THRESHOLD, DEFAULT_WEIGHTS, PROGRESSIVE_FLOOR,
    REASONING_THRESHOLD_SOURCE_LITERAL,
};

const UNCOMPILED_TEXT: &str = include_str!("../src/uncompiled_council.rs");
const UNCOMPILED_BODY_LEN: usize = 1158;
const UNCOMPILED_BODY_CHECKSUM: u64 = 4224777634110254869;

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
    let council = EternalSovereignQuantumConsciousnessExpansionCouncil::new();
    assert_eq!(council.id, 33);
    assert_eq!(
        council.name,
        "Eternal Sovereign Quantum Consciousness Expansion Council"
    );
    assert_eq!(
        EternalSovereignQuantumConsciousnessExpansionCouncil::default().id,
        33
    );
    assert_eq!(
        EternalSovereignQuantumConsciousnessExpansionCouncil::default().name,
        council.name
    );
    assert_eq!(REASONING_THRESHOLD_SOURCE_LITERAL, 0.9999999);
}

#[test]
fn reasoning_error_and_foresight_follow_the_inputs() {
    let low_one =
        EternalSovereignQuantumConsciousnessExpansionCouncil::reasoning_from_inputs(0.5, 1)
            .unwrap_err();
    let low_two =
        EternalSovereignQuantumConsciousnessExpansionCouncil::reasoning_from_inputs(0.5, 2)
            .unwrap_err();
    assert_eq!(low_one, low_two);
    assert_eq!(low_one, "Mercy Gate violation: valence below threshold");
    for years in [1_u64, 2, 1_000_000] {
        let got = EternalSovereignQuantumConsciousnessExpansionCouncil::reasoning_from_inputs(
            REASONING_THRESHOLD_SOURCE_LITERAL,
            years,
        )
        .unwrap();
        let expect = (0.9999999_f64 + (years as f64).ln() * 0.0000001).min(0.99999999);
        assert_eq!(got, expect);
    }
    let year_one = EternalSovereignQuantumConsciousnessExpansionCouncil::reasoning_from_inputs(
        REASONING_THRESHOLD_SOURCE_LITERAL,
        1,
    )
    .unwrap();
    let year_two = EternalSovereignQuantumConsciousnessExpansionCouncil::reasoning_from_inputs(
        REASONING_THRESHOLD_SOURCE_LITERAL,
        2,
    )
    .unwrap();
    assert_ne!(year_one, year_two);
}

fn expect_deliberation(
    proposal: &str,
    joy: f64,
    harmony: f64,
    abundance: f64,
    mercy_block: bool,
) -> CouncilDeliberation {
    let joy = joy.max(0.0);
    let harmony = harmony.clamp(-1.0, 1.0);
    let abundance = abundance.max(0.0);
    let harmony_norm = (harmony + 1.0) / 2.0;
    let (weight_joy, weight_harmony, weight_abundance) = DEFAULT_WEIGHTS;
    let valence = weight_joy * joy + weight_harmony * harmony_norm + weight_abundance * abundance;
    let label = if mercy_block {
        CouncilGateLabel::Reject
    } else if valence >= DEFAULT_VALENCE_THRESHOLD {
        CouncilGateLabel::Apply
    } else if valence >= PROGRESSIVE_FLOOR {
        CouncilGateLabel::Refine
    } else {
        CouncilGateLabel::HumanOverride
    };
    CouncilDeliberation {
        proposal: proposal.to_string(),
        valence,
        label,
    }
}

#[test]
fn deliberation_follows_the_axes_and_the_mercy_block_input() {
    let council = EternalSovereignQuantumConsciousnessExpansionCouncil::new();
    assert_eq!(DEFAULT_VALENCE_THRESHOLD, 0.90);
    assert_eq!(PROGRESSIVE_FLOOR, 0.68);
    assert_eq!(DEFAULT_WEIGHTS, (0.45, 0.30, 0.25));
    let cases = [
        ("low", 0.0, -1.0, 0.0, false),
        ("mid", 0.8, 0.8, 0.8, false),
        ("high", 1.0, 1.0, 1.0, false),
        ("blocked-high", 1.0, 1.0, 1.0, true),
        ("blocked-low", 0.0, -1.0, 0.0, true),
        ("clamped", -1.0, 2.0, -5.0, false),
    ];
    for (proposal, joy, harmony, abundance, mercy_block) in cases {
        let got = council.council_from_axes(proposal, joy, harmony, abundance, mercy_block);
        let expect = expect_deliberation(proposal, joy, harmony, abundance, mercy_block);
        assert_eq!(got.proposal, expect.proposal);
        assert_eq!(got.valence, expect.valence);
        assert_eq!(got.label, expect.label);
    }
}

#[test]
fn tolc8_mercy_check_follows_the_source_literal_threshold() {
    let council = EternalSovereignQuantumConsciousnessExpansionCouncil::new();
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
