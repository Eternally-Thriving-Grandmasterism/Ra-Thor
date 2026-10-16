//! Exact values for the compiled council wire.
//! Threshold figures are source literals.

use std::fs;
use std::path::Path;

use eternal_sovereign_infinite_horizon_council::{
    EternalSovereignInfiniteHorizonCouncil, REASONING_THRESHOLD_SOURCE_LITERAL,
};

const UNCOMPILED_TEXT: &str = include_str!("../src/uncompiled_council.rs");
const UNCOMPILED_BODY_LEN: usize = 1052;
const UNCOMPILED_BODY_CHECKSUM: u64 = 15728484574158329081;

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
    let council = EternalSovereignInfiniteHorizonCouncil::new();
    assert_eq!(council.id, 31);
    assert_eq!(council.name, "Eternal Sovereign Infinite Horizon Council");
    assert_eq!(EternalSovereignInfiniteHorizonCouncil::default().id, 31);
    assert_eq!(
        EternalSovereignInfiniteHorizonCouncil::default().name,
        council.name
    );
    assert_eq!(REASONING_THRESHOLD_SOURCE_LITERAL, 0.9999999);
}

#[test]
fn reasoning_error_and_foresight_follow_the_inputs() {
    let low_one =
        EternalSovereignInfiniteHorizonCouncil::reasoning_from_inputs(0.5, 1).unwrap_err();
    let low_two =
        EternalSovereignInfiniteHorizonCouncil::reasoning_from_inputs(0.5, 2).unwrap_err();
    assert_eq!(low_one, low_two);
    assert_eq!(low_one, "Mercy Gate violation: valence below threshold");
    for years in [1_u64, 2, 1_000_000] {
        let got = EternalSovereignInfiniteHorizonCouncil::reasoning_from_inputs(
            REASONING_THRESHOLD_SOURCE_LITERAL,
            years,
        )
        .unwrap();
        let expect = (0.9999999_f64 + (years as f64).ln() * 0.0000001).min(0.99999999);
        assert_eq!(got, expect);
    }
    let year_one = EternalSovereignInfiniteHorizonCouncil::reasoning_from_inputs(
        REASONING_THRESHOLD_SOURCE_LITERAL,
        1,
    )
    .unwrap();
    let year_two = EternalSovereignInfiniteHorizonCouncil::reasoning_from_inputs(
        REASONING_THRESHOLD_SOURCE_LITERAL,
        2,
    )
    .unwrap();
    assert_ne!(year_one, year_two);
}

#[test]
#[allow(clippy::bool_assert_comparison)] // expected values are the literals true and false
fn tolc8_mercy_check_follows_the_source_literal_threshold() {
    let council = EternalSovereignInfiniteHorizonCouncil::new();
    let threshold = council.valence_threshold();
    assert_eq!(threshold, 0.9999999);
    let below = f64::from_bits(threshold.to_bits() - 1);
    let above = f64::from_bits(threshold.to_bits() + 1);
    assert!(below < threshold);
    assert!(above > threshold);
    // Literal results. `below` is under the source literal, so the check is false.
    assert_eq!(council.tolc8_mercy_check(below), false);
    assert_eq!(council.tolc8_mercy_check(threshold), true);
    assert_eq!(council.tolc8_mercy_check(above), true);
}
