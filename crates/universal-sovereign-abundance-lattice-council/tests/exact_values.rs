//! Exact values for the compiled council wire.
//! Threshold figures are source literals.

use std::fs;
use std::path::Path;

use universal_sovereign_abundance_lattice_council::{
    UniversalSovereignAbundanceLatticeCouncil, REASONING_THRESHOLD_SOURCE_LITERAL,
    UNCOMPILED_BODY_CHECKSUM, UNCOMPILED_BODY_LEN,
};

fn preserved_body() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/uncompiled_council.rs");
    let text = fs::read_to_string(path).unwrap();
    let marker = "#![rustfmt::skip]\n";
    let (head, body) = text.split_once(marker).unwrap();
    assert!(head.starts_with(
        "//! Design-only (STUB-AUDIT-1 REASONING-COUNCIL-WIRES-1 council note): not compiled;",
    ),);
    assert_eq!(text.lines().nth(1), Some("#![rustfmt::skip]"));
    body.to_string()
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
    assert_eq!(body_checksum(&body), UNCOMPILED_BODY_CHECKSUM);
    let lib = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs")).unwrap();
    assert_eq!(lib.matches("mod uncompiled_council").count(), 0);
}

#[test]
fn record_stores_the_historical_title() {
    let council = UniversalSovereignAbundanceLatticeCouncil::new();
    assert_eq!(council.id, 28);
    assert_eq!(
        council.name,
        "Universal Sovereign Abundance Lattice Council"
    );
    assert_eq!(UniversalSovereignAbundanceLatticeCouncil::default().id, 28);
    assert_eq!(
        UniversalSovereignAbundanceLatticeCouncil::default().name,
        council.name
    );
    assert_eq!(REASONING_THRESHOLD_SOURCE_LITERAL, 0.9999999);
}

#[test]
fn reasoning_error_and_foresight_follow_the_inputs() {
    let low_one =
        UniversalSovereignAbundanceLatticeCouncil::reasoning_from_inputs(0.5, 1).unwrap_err();
    let low_two =
        UniversalSovereignAbundanceLatticeCouncil::reasoning_from_inputs(0.5, 2).unwrap_err();
    assert_eq!(low_one, low_two);
    assert_eq!(low_one, "Mercy Gate violation: valence below threshold");
    for years in [1_u64, 2, 1_000_000] {
        let got = UniversalSovereignAbundanceLatticeCouncil::reasoning_from_inputs(
            REASONING_THRESHOLD_SOURCE_LITERAL,
            years,
        )
        .unwrap();
        let expect = (0.9999999_f64 + (years as f64).ln() * 0.0000001).min(0.99999999);
        assert_eq!(got, expect);
    }
    let year_one = UniversalSovereignAbundanceLatticeCouncil::reasoning_from_inputs(
        REASONING_THRESHOLD_SOURCE_LITERAL,
        1,
    )
    .unwrap();
    let year_two = UniversalSovereignAbundanceLatticeCouncil::reasoning_from_inputs(
        REASONING_THRESHOLD_SOURCE_LITERAL,
        2,
    )
    .unwrap();
    assert_ne!(year_one, year_two);
}
