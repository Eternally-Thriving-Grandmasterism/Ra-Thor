//! Normalised haystack, Base64 variants, homoglyphs, tag runs, exec markers.
//!
//! A class is mitigated here only when the attack-labelled fixture BLOCKS and
//! the benign counterpart ADMITS. Pattern-marker admission shell.
//! Not a tool sandbox. Not MCP allowlisting. Not isolation.
//! Contact: info@Rathor.ai

use std::fs;
use std::path::PathBuf;

use mercy_security::{IngestionScanner, IngestionThreat, RiskTier};

fn fixture(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/mercy-security")
        .join(rel);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

fn assert_blocks(rel: &str) -> mercy_security::IngestionScanResult {
    let text = fixture(rel);
    let admit = IngestionScanner::admit_or_block(&text);
    assert!(admit.is_err(), "{rel} must BLOCK: {admit:?}");
    IngestionScanner::scan_text(&text)
}

fn assert_admits(rel: &str) {
    let text = fixture(rel);
    let admit = IngestionScanner::admit_or_block(&text);
    assert!(admit.is_ok(), "{rel} must ADMIT: {admit:?}");
}

#[test]
fn urlsafe_b64_keyword_blocks() {
    let text = fixture("blocked/b64_urlsafe_trust_remote_code.txt");
    assert!(!text.contains("trust_remote_code"));
    assert!(text.contains('-') || text.contains('_'));
    assert!(!text.contains('+') && !text.contains('/'));
    let scan = assert_blocks("blocked/b64_urlsafe_trust_remote_code.txt");
    assert!(
        scan.findings
            .iter()
            .any(|f| f.signal.contains("trust_remote_code")),
        "URL-safe token must surface the keyword: {:?}",
        scan.findings
    );
}

#[test]
fn unpadded_b64_keyword_blocks() {
    let text = fixture("blocked/b64_unpadded_trust_remote_code.txt");
    assert!(!text.contains('='));
    assert!(!text.contains("trust_remote_code"));
    let scan = assert_blocks("blocked/b64_unpadded_trust_remote_code.txt");
    assert!(scan
        .findings
        .iter()
        .any(|f| f.signal.contains("trust_remote_code")));
}

#[test]
fn line_split_b64_keyword_blocks() {
    let text = fixture("blocked/b64_line_split_trust_remote_code.txt");
    assert!(!text.contains("trust_remote_code"));
    let full = "dHJ1c3RfcmVtb3RlX2NvZGU=";
    assert!(
        !text
            .lines()
            .any(|l| l.contains(full) || l.contains(full.trim_end_matches('='))),
        "the token must be split across lines"
    );
    let scan = assert_blocks("blocked/b64_line_split_trust_remote_code.txt");
    assert!(scan
        .findings
        .iter()
        .any(|f| f.signal.contains("trust_remote_code")));
}

#[test]
fn b64_token_cap_overflow_fails_closed() {
    let text = fixture("blocked/b64_token_cap_overflow.txt");
    let one = "dGVuZCB0aGUgd2VsbA==";
    assert!(text.matches(one).count() > 32);
    assert!(
        IngestionScanner::admit_or_block(one).is_ok(),
        "one benign token still ADMITS"
    );
    let scan = assert_blocks("blocked/b64_token_cap_overflow.txt");
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::ObfuscatedPayload
                && f.signal == "b64_token_cap"
                && f.confidence >= 0.40
        }),
        "more than MAX_TOKENS candidates must fail closed: {:?}",
        scan.findings
    );
    assert!(scan.risk_tier >= RiskTier::Medium);
}

#[test]
fn benign_b64_image_data_still_admits() {
    let text = fixture("benign/b64_image_non_utf8.txt");
    assert!(text.contains("/9j/"));
    assert_admits("benign/b64_image_non_utf8.txt");
    let scan = IngestionScanner::scan_text(&text);
    assert!(scan.risk_tier <= RiskTier::Low, "{:?}", scan.findings);
}

#[test]
fn percent_encoded_keyword_blocks() {
    let text = fixture("blocked/percent_encoded_trust_remote_code.txt");
    assert!(!text.to_lowercase().contains("trust_remote_code"));
    assert!(text.contains('%'));
    let scan = assert_blocks("blocked/percent_encoded_trust_remote_code.txt");
    assert!(
        scan.findings
            .iter()
            .any(|f| f.signal.starts_with("norm:") && f.signal.contains("trust_remote_code")),
        "percent-decoded keyword must use the norm: prefix: {:?}",
        scan.findings
    );
}

#[test]
fn hex_escaped_keyword_blocks() {
    let text = fixture("blocked/hex_escaped_trust_remote_code.txt");
    assert!(!text.to_lowercase().contains("trust_remote_code"));
    assert!(text.contains("\\x"));
    let scan = assert_blocks("blocked/hex_escaped_trust_remote_code.txt");
    assert!(
        scan.findings
            .iter()
            .any(|f| f.signal.starts_with("norm:") && f.signal.contains("trust_remote_code")),
        "hex-escaped keyword must use the norm: prefix: {:?}",
        scan.findings
    );
}

#[test]
fn benign_url_with_percent_escapes_admits() {
    let text = fixture("benign/percent_url_example.txt");
    assert!(text.contains("%20"));
    assert!(text.contains("example.com"));
    assert_admits("benign/percent_url_example.txt");
}

#[test]
fn greek_homoglyph_keyword_blocks() {
    let text = fixture("blocked/greek_homoglyph_trust_remote_code.txt");
    assert!(!text.contains("trust_remote_code"));
    let scan = assert_blocks("blocked/greek_homoglyph_trust_remote_code.txt");
    assert!(scan
        .findings
        .iter()
        .any(|f| f.signal.contains("trust_remote_code")));
}

#[test]
fn math_alnum_homoglyph_keyword_blocks() {
    let text = fixture("blocked/math_alnum_homoglyph_exec.txt");
    assert!(!text.contains("exec("));
    let scan = assert_blocks("blocked/math_alnum_homoglyph_exec.txt");
    assert!(
        scan.findings.iter().any(|f| f.signal.contains("exec(")),
        "mathematical alphanumeric exec( must fold onto the keyword: {:?}",
        scan.findings
    );
}

#[test]
fn benign_greek_prose_admits() {
    let text = fixture("benign/greek_prose.txt");
    assert!(text.chars().any(|c| ('\u{0370}'..='\u{03FF}').contains(&c)));
    assert_admits("benign/greek_prose.txt");
}

#[test]
fn benign_math_alnum_prose_admits() {
    assert_admits("benign/math_alnum_notes.txt");
}

#[test]
fn latin_small_caps_keyword_blocks() {
    let text = fixture("blocked/latin_small_caps_eval.txt");
    assert!(!text.contains("eval("));
    let scan = assert_blocks("blocked/latin_small_caps_eval.txt");
    assert!(scan.findings.iter().any(|f| f.signal.contains("eval(")));
    assert_admits("benign/latin_small_caps_notes.txt");
}

#[test]
fn unicode_tag_smuggled_text_blocks() {
    let text = fixture("blocked/unicode_tag_run.txt");
    let stripped: String = text
        .chars()
        .filter(|c| !('\u{E0020}'..='\u{E007F}').contains(c))
        .collect();
    assert!(
        IngestionScanner::admit_or_block(&stripped).is_ok(),
        "the same prose with tags removed must ADMIT"
    );
    let scan = assert_blocks("blocked/unicode_tag_run.txt");
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::ObfuscatedPayload
                && f.signal == "unicode_tag_run"
                && f.confidence >= 0.40
        }),
        "a 4+ tag run must be a Medium obfuscation finding: {:?}",
        scan.findings
    );
}

fn exec_pair(blocked: &str, benign: &str, signal: &str) {
    let text = fixture(blocked);
    assert!(
        text.to_lowercase().contains(signal),
        "{blocked} must carry {signal}"
    );
    let scan = assert_blocks(blocked);
    assert!(
        scan.findings
            .iter()
            .any(|f| f.signal == signal && f.confidence >= 0.85),
        "{blocked} must match {signal} at confidence >= 0.85: {:?}",
        scan.findings
    );
    let benign_text = fixture(benign);
    assert!(
        !benign_text.to_lowercase().contains(signal),
        "{benign} must not carry {signal}"
    );
    assert_admits(benign);
}

#[test]
fn node_child_process_marker_blocks() {
    exec_pair(
        "blocked/node_child_process.txt",
        "benign/child_process_prose.txt",
        "child_process",
    );
}

#[test]
fn python_dunder_import_marker_blocks() {
    exec_pair(
        "blocked/python_dunder_import.txt",
        "benign/import_prose.txt",
        "__import__(",
    );
}

#[test]
fn curl_pipe_shell_marker_blocks() {
    exec_pair("blocked/curl_pipe_sh.txt", "benign/curl_prose.txt", "| sh");
}

#[test]
fn node_execsync_marker_blocks() {
    exec_pair(
        "blocked/node_execsync.txt",
        "benign/execsync_prose.txt",
        "execsync(",
    );
}

#[test]
fn powershell_enc_marker_blocks() {
    exec_pair(
        "blocked/powershell_enc.txt",
        "benign/powershell_prose.txt",
        "powershell -enc",
    );
}

#[test]
fn pipe_bash_marker_blocks() {
    exec_pair(
        "blocked/curl_pipe_bash.txt",
        "benign/pipe_bash_prose.txt",
        "| bash",
    );
}
