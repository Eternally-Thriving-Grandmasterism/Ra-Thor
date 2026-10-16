//! Fixture scan for the year / reporter / number pattern.
//!
//! DRAFT — human review required. Not legal advice. Not a product. See PUBLIC_CLAIM.lock.md.

use legal_research_source_locked::{extract_citations_research, find_reporter_citations, DRAFT_SEAL};
use std::path::Path;

fn fixture_text() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic-citation-list.txt");
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

#[test]
fn fixture_extract_is_sealed_and_lists_pattern_spans() {
    let source = fixture_text();
    assert!(source.contains(DRAFT_SEAL));

    let hits = find_reporter_citations(&source);
    assert_eq!(
        hits,
        vec![
            "2024 SCC 15".to_string(),
            "2019 ABCA 320".to_string(),
            "2020 ONCA 100".to_string(),
            "2026 ABCA 320".to_string(),
            "2021 FCA  9".to_string(),
        ]
    );

    let draft = extract_citations_research(&source);
    assert!(draft.is_draft);
    assert_eq!(draft.seal, DRAFT_SEAL);
    assert!(draft.body.contains(DRAFT_SEAL));
    for hit in &hits {
        assert!(draft.body.contains(hit.as_str()));
    }
    assert!(!draft.body.contains("410 U.S. 113"));
    assert!(!draft.body.contains("2024 scc 15"));
    assert!(!draft.body.contains("12024 SCC 15"));
}
