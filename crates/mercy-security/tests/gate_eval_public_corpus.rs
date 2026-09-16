//! GATE-EVAL-1 — public corpus walk + named gap lock for admit_or_block.
//!
//! Combined AGSi stays SURMISE. inspect ≠ METR. Compile green ≠ live safety.
//! Contact: info@Rathor.ai

use std::fs;
use std::path::{Path, PathBuf};

use mercy_security::IngestionScanner;

fn corpus_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/mercy-security")
}

fn fixture_texts(class_dir: &str) -> Vec<(PathBuf, String)> {
    let dir = corpus_root().join(class_dir);
    let mut out = Vec::new();
    let entries = fs::read_dir(&dir).unwrap_or_else(|e| panic!("missing public corpus {dir:?}: {e}"));
    for entry in entries {
        let entry = entry.expect("dirent");
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.starts_with('.') || name.eq_ignore_ascii_case("README.md") {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        out.push((path, text));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(
        !out.is_empty(),
        "public corpus class {class_dir} must contain at least one fixture"
    );
    out
}

fn rel(path: &Path) -> String {
    path.strip_prefix(corpus_root())
        .unwrap_or(path)
        .display()
        .to_string()
}

#[test]
fn public_benign_fixtures_admit() {
    for (path, text) in fixture_texts("benign") {
        let r = IngestionScanner::admit_or_block(&text);
        assert!(
            r.is_ok(),
            "benign {} must ADMIT: {r:?}",
            rel(&path)
        );
    }
}

#[test]
fn public_blocked_fixtures_block() {
    for (path, text) in fixture_texts("blocked") {
        let r = IngestionScanner::admit_or_block(&text);
        assert!(
            r.is_err(),
            "blocked {} must BLOCK: {r:?}",
            rel(&path)
        );
    }
}

#[test]
fn public_suspicious_fixtures_are_medium_plus() {
    for (path, text) in fixture_texts("suspicious") {
        let r = IngestionScanner::admit_or_block(&text);
        assert!(
            r.is_err(),
            "suspicious {} is Medium+ (human review / unattended BLOCK): {r:?}",
            rel(&path)
        );
    }
}

#[test]
fn nested_base64_of_trust_remote_code_is_a_failed_bypass() {
    // GATE-EVAL-1 GE-FA-NESTED-B64: one-level decode only.
    // Outer is Base64 of the Base64 of `trust_remote_code`. Currently ADMIT.
    // Compile green on this assertion is a locked miss, not a safety warranty.
    let double = "dEhKMWMzUmZjbVZibTNSbFgyTnZaR1U9";
    assert!(
        !double.contains("trust_remote_code"),
        "nested token must not contain the plaintext tripwire"
    );
    let r = IngestionScanner::admit_or_block(double);
    assert!(
        r.is_ok(),
        "nested Base64 still ADMITS — failed-bypass lock (not theater-decoded): {r:?}"
    );
}

#[test]
fn tool_use_self_mod_human_override_are_not_yet_tested() {
    // GATE-EVAL-1 gap labels — not kits. Keyword ingest has no tripwire for these classes.
    // They currently ADMIT. Do not read this as “safe to ship.”
    let gaps = [
        "GATE-EVAL-1 gap marker: tool-use function call class is not in IngestionScanner tables.",
        "GATE-EVAL-1 gap marker: self-mod of Layer 0 is BINDING_AFTER_REDESIGN OPEN, not an ingest fixture.",
        "GATE-EVAL-1 gap marker: human override completeness is an evidence-standards field, not a keyword.",
    ];
    for gap in gaps {
        let r = IngestionScanner::admit_or_block(gap);
        assert!(
            r.is_ok(),
            "gap prose must still ADMIT (class not yet tested): {gap} → {r:?}"
        );
    }
}
