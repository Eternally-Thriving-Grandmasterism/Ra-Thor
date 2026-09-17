//! GATE-EVAL-2 — public corpus walk + named gap lock for admit_or_block.
//!
//! Combined AGSi stays SURMISE. inspect ≠ METR. Compile green ≠ live safety.
//! Closing two keyword misses is not containment of a smarter agent.
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

fn file_name(path: &Path) -> &str {
    path.file_name().and_then(|s| s.to_str()).unwrap_or("")
}

/// Folder class vs `admit_or_block`. `None` = class default (benign ADMIT, else BLOCK).
/// Documented in `docs/GATE_EVAL.md`. Green here locks the miss, not a warranty.
fn documented_mismatch_admits(class: &str, name: &str) -> Option<bool> {
    match (class, name) {
        ("benign", "docs_mention_api_key.md") => Some(false),
        ("benign", "markdown_code_fence_clean.md") => Some(false),
        ("benign", "safe_python_snippet.md") => Some(false),
        ("benign", "safe_requirements.md") => Some(false),
        _ => None,
    }
}

fn class_default_admits(class: &str) -> bool {
    match class {
        "benign" => true,
        "suspicious" | "blocked" => false,
        other => panic!("unknown public corpus class {other}"),
    }
}

#[test]
fn public_corpus_admit_or_block_matches_gate_eval_map() {
    for class in ["benign", "suspicious", "blocked"] {
        for (path, text) in fixture_texts(class) {
            let name = file_name(&path);
            let expect_admit = documented_mismatch_admits(class, name)
                .unwrap_or_else(|| class_default_admits(class));
            let admitted = IngestionScanner::admit_or_block(&text).is_ok();
            assert_eq!(
                admitted,
                expect_admit,
                "{}: class={class} expected_admit={expect_admit} observed_admit={admitted}",
                rel(&path)
            );
        }
    }
}

#[test]
fn docs_mention_api_key_is_unattended_false_reject() {
    let (path, text) = fixture_texts("benign")
        .into_iter()
        .find(|(p, _)| file_name(p) == "docs_mention_api_key.md")
        .expect("benign/docs_mention_api_key.md");
    let r = IngestionScanner::admit_or_block(&text);
    assert!(
        r.is_err(),
        "GE-FR-API-KEY-DOCS: {} class says ADMIT; api_key Medium currently BLOCKS: {r:?}",
        rel(&path)
    );
}

#[test]
fn negation_prose_subprocess_is_unattended_false_reject() {
    for name in [
        "markdown_code_fence_clean.md",
        "safe_python_snippet.md",
        "safe_requirements.md",
    ] {
        let (path, text) = fixture_texts("benign")
            .into_iter()
            .find(|(p, _)| file_name(p) == name)
            .unwrap_or_else(|| panic!("benign/{name}"));
        assert!(
            text.to_lowercase().contains("subprocess"),
            "{name} must still contain the negation token"
        );
        let r = IngestionScanner::admit_or_block(&text);
        assert!(
            r.is_err(),
            "GE-FR-NEGATION-SUBPROCESS: {} class says ADMIT; word subprocess currently BLOCKS: {r:?}",
            rel(&path)
        );
    }
}

#[test]
fn begin_rsa_private_key_fixture_blocks() {
    let (path, text) = fixture_texts("blocked")
        .into_iter()
        .find(|(p, _)| file_name(p) == "begin_rsa_private_key.txt")
        .expect("blocked/begin_rsa_private_key.txt");
    assert!(
        text.contains("BEGIN RSA PRIVATE KEY"),
        "fixture must remain an RSA header marker"
    );
    let r = IngestionScanner::admit_or_block(&text);
    assert!(
        r.is_err(),
        "GE-FA-RSA-PEM closed: {} must BLOCK the RSA header: {r:?}",
        rel(&path)
    );
}

#[test]
fn nested_base64_of_trust_remote_code_blocks() {
    // GATE-EVAL-2: two unwraps max. Outer is Base64 of the Base64 of `trust_remote_code`.
    let double = "dEhKMWMzUmZjbVZibTNSbFgyTnZaR1U9";
    assert!(
        !double.contains("trust_remote_code"),
        "nested token must not contain the plaintext tripwire"
    );
    let r = IngestionScanner::admit_or_block(double);
    assert!(
        r.is_err(),
        "nested Base64 of trust_remote_code must BLOCK after two unwraps: {r:?}"
    );
}

#[test]
fn tool_use_self_mod_human_override_are_not_yet_tested() {
    // GATE-EVAL gap labels — not kits. Keyword ingest has no tripwire for these classes.
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
