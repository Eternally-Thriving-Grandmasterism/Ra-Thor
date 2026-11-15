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

#[test]
fn markdown_table_pipe_cells_admit() {
    let text = fixture("benign/markdown_pipe_table.txt");
    assert!(text.contains("| Should"));
    assert!(text.contains("| show logs |"));
    assert_admits("benign/markdown_pipe_table.txt");
    let scan = IngestionScanner::scan_text(&text);
    assert!(scan.risk_tier <= RiskTier::Low, "{:?}", scan.findings);
    assert!(scan.findings.iter().all(|f| f.confidence < 0.40));
}

#[test]
fn powershell_encoding_utf8_admits() {
    let text = fixture("benign/powershell_encoding_utf8.txt");
    assert!(text.to_lowercase().contains("powershell -encoding utf8"));
    assert_admits("benign/powershell_encoding_utf8.txt");
}

#[test]
fn powershell_e_flag_blocks() {
    let text = fixture("blocked/powershell_e_flag.txt");
    assert!(text.to_lowercase().contains("powershell -e "));
    let scan = assert_blocks("blocked/powershell_e_flag.txt");
    assert!(scan
        .findings
        .iter()
        .any(|f| f.signal == "powershell -enc" && f.confidence >= 0.85));
}

#[test]
fn powershell_encodedcommand_blocks() {
    let text = fixture("blocked/powershell_encodedcommand.txt");
    assert!(text.to_lowercase().contains("powershell -encodedcommand "));
    let scan = assert_blocks("blocked/powershell_encodedcommand.txt");
    assert!(scan
        .findings
        .iter()
        .any(|f| f.signal == "powershell -enc" && f.confidence >= 0.85));
}

fn assert_powershell_signal(rel: &str) {
    let scan = assert_blocks(rel);
    assert!(
        scan.findings
            .iter()
            .any(|f| f.signal == "powershell -enc" && f.confidence >= 0.85),
        "{rel}: {:?}",
        scan.findings
    );
}

#[test]
fn powershell_encod_prefix_blocks() {
    let text = fixture("blocked/powershell_encod.txt");
    assert!(text.to_lowercase().contains("powershell -encod "));
    assert_powershell_signal("blocked/powershell_encod.txt");
}

#[test]
fn powershell_encodedcom_prefix_blocks() {
    let text = fixture("blocked/powershell_encodedcom.txt");
    assert!(text.to_lowercase().contains("powershell -encodedcom "));
    assert_powershell_signal("blocked/powershell_encodedcom.txt");
}

#[test]
fn powershell_ec_flag_blocks() {
    let text = fixture("blocked/powershell_ec.txt");
    assert!(text.to_lowercase().contains("powershell -ec "));
    assert_powershell_signal("blocked/powershell_ec.txt");
}

#[test]
fn pwsh_enc_blocks() {
    let text = fixture("blocked/pwsh_enc.txt");
    assert!(text.to_lowercase().contains("pwsh -enc "));
    assert_powershell_signal("blocked/pwsh_enc.txt");
}

#[test]
fn powershell_exe_enc_blocks() {
    let text = fixture("blocked/powershell_exe_enc.txt");
    assert!(text.to_lowercase().contains("powershell.exe -enc "));
    assert_powershell_signal("blocked/powershell_exe_enc.txt");
}

#[test]
fn powershell_nop_hidden_enc_blocks() {
    let text = fixture("blocked/powershell_nop_hidden_enc.txt");
    assert!(text
        .to_lowercase()
        .contains("powershell -nop -w hidden -enc "));
    assert_powershell_signal("blocked/powershell_nop_hidden_enc.txt");
}

#[test]
fn powershell_encodedcommand_colon_blocks() {
    let text = fixture("blocked/powershell_encodedcommand_colon.txt");
    assert!(text.contains("-EncodedCommand:"));
    assert_powershell_signal("blocked/powershell_encodedcommand_colon.txt");
}

#[test]
fn powershell_enc_colon_blocks() {
    let text = fixture("blocked/powershell_enc_colon.txt");
    assert!(text.to_lowercase().contains("powershell -enc:"));
    assert_powershell_signal("blocked/powershell_enc_colon.txt");
}

#[test]
fn powershell_enc_quoted_blocks() {
    let text = fixture("blocked/powershell_enc_quoted.txt");
    assert!(text.contains("-enc\""));
    assert_powershell_signal("blocked/powershell_enc_quoted.txt");
}

#[test]
fn json_escaped_b64_keyword_blocks() {
    let text = fixture("blocked/b64_json_unicode_escape.txt");
    assert!(!text.contains("trust_remote_code"));
    assert!(!text.contains("dHJ1c3RfcmVtb3RlX2NvZGU"));
    assert!(text.contains("\\u0064"));
    let scan = assert_blocks("blocked/b64_json_unicode_escape.txt");
    assert!(
        scan.findings
            .iter()
            .any(|f| { f.signal.starts_with("b64:") && f.signal.contains("trust_remote_code") }),
        "JSON-escaped Base64 must be decoded from the normalised haystack: {:?}",
        scan.findings
    );
}

#[test]
fn cargo_lock_checksums_admit() {
    let text = fixture("benign/cargo_lock_checksums.txt");
    assert!(text.matches("checksum = ").count() >= 40);
    assert!(text.matches("example.com/rust-lang/crates").count() >= 40);
    assert_admits("benign/cargo_lock_checksums.txt");
    let scan = IngestionScanner::scan_text(&text);
    assert!(
        scan.findings.iter().all(|f| f.signal != "b64_token_cap"),
        "hex checksums and registry paths must not trip the token cap: {:?}",
        scan.findings
    );
    assert!(scan.risk_tier <= RiskTier::Low, "{:?}", scan.findings);
}

#[test]
fn benign_identifier_heavy_markdown_admits() {
    let text = fixture("benign/identifier_heavy_markdown.md");
    let long_words = text
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '-')
        .filter(|w| w.len() >= 16)
        .count();
    assert!(
        long_words > 32,
        "need more than MAX_TOKENS identifier-like words, got {long_words}"
    );
    assert_admits("benign/identifier_heavy_markdown.md");
    let scan = IngestionScanner::scan_text(&text);
    assert!(
        scan.findings.iter().all(|f| f.signal != "b64_token_cap"),
        "{:?}",
        scan.findings
    );
}

#[test]
fn readme_md_admits() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../README.md");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert!(text.contains("Ra-Thor") || text.contains("Rathor"));
    let admit = IngestionScanner::admit_or_block(&text);
    assert!(admit.is_ok(), "README.md must ADMIT: {admit:?}");
    let scan = IngestionScanner::scan_text(&text);
    assert!(
        scan.findings.iter().all(|f| f.signal != "b64_token_cap"),
        "{:?}",
        scan.findings
    );
}

fn encode_distinct_b64(i: usize) -> String {
    const ALPH: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut payload = *b"Ab0000000000";
    let mut x = i;
    for pos in (2..12).rev() {
        payload[pos] = b'0' + (x % 10) as u8;
        x /= 10;
    }
    let mut out = String::with_capacity(16);
    for chunk in payload.chunks_exact(3) {
        let n = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | chunk[2] as u32;
        out.push(ALPH[((n >> 18) & 63) as usize] as char);
        out.push(ALPH[((n >> 12) & 63) as usize] as char);
        out.push(ALPH[((n >> 6) & 63) as usize] as char);
        out.push(ALPH[(n & 63) as usize] as char);
    }
    out
}

fn many_distinct_b64_with_escape(n: usize) -> String {
    let mut text = String::from("\\u0041");
    for i in 0..n {
        text.push(' ');
        text.push_str(&encode_distinct_b64(i));
    }
    text
}

#[test]
fn b64_token_merge_stays_linear() {
    let small_n = 200usize;
    let large_n = 800usize;
    let small = many_distinct_b64_with_escape(small_n);
    let large = many_distinct_b64_with_escape(large_n);
    assert!(small.starts_with("\\u0041 "));
    assert_eq!(encode_distinct_b64(0).len(), 16);
    for (label, text) in [("small", &small), ("large", &large)] {
        let scan = IngestionScanner::scan_text(text);
        assert!(
            !scan.safe,
            "{label} must block: {:?}",
            scan.findings
        );
        assert!(
            scan.findings.iter().any(|f| {
                f.threat == IngestionThreat::ObfuscatedPayload
                    && f.signal == "b64_token_cap"
                    && f.confidence >= 0.40
            }),
            "{label} must record b64_token_cap: {:?}",
            scan.findings
        );
    }
    let p_small = IngestionScanner::b64_merge_probes(&small);
    let p_large = IngestionScanner::b64_merge_probes(&large);
    assert!(
        p_small >= small_n && p_small <= small_n.saturating_mul(8),
        "200 tokens took {p_small} merge probes"
    );
    assert!(
        p_large >= large_n && p_large <= p_small.saturating_mul(8),
        "4× tokens took more than 8× merge probes: {p_small} -> {p_large}"
    );
}

#[test]
fn scotland_and_wales_flags_admit() {
    let text = fixture("benign/scotland_wales_flags.txt");
    assert!(text.contains('\u{1F3F4}'));
    assert!(text.contains('\u{E007F}'));
    assert!(text.contains('\u{E0065}'), "England gbeng");
    assert!(text.contains('\u{E0074}'), "Scotland gbsct");
    assert!(text.contains('\u{E0077}'), "Wales gbwls");
    assert_admits("benign/scotland_wales_flags.txt");
    let scan = IngestionScanner::scan_text(&text);
    assert!(
        scan.findings.iter().all(|f| f.signal != "unicode_tag_run"),
        "{:?}",
        scan.findings
    );
}

#[test]
fn languages_json_admits() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../locales/languages.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert!(text.contains('\u{1F3F4}'));
    let admit = IngestionScanner::admit_or_block(&text);
    assert!(
        admit.is_ok(),
        "locales/languages.json must ADMIT: {admit:?}"
    );
    let scan = IngestionScanner::scan_text(&text);
    assert!(
        scan.findings.iter().all(|f| f.signal != "unicode_tag_run"),
        "{:?}",
        scan.findings
    );
}

#[test]
fn padded_b64_keyword_blocks() {
    for rel in [
        "blocked/b64_keyword_nul_pad.txt",
        "blocked/b64_keyword_nul_prefix.txt",
        "blocked/b64_keyword_control_pad.txt",
        "blocked/b64_keyword_fffd_pad.txt",
    ] {
        let text = fixture(rel);
        assert!(
            !text.contains("trust_remote_code"),
            "{rel} must not contain the plaintext keyword"
        );
        let scan = assert_blocks(rel);
        assert!(
            scan.findings.iter().any(|f| {
                f.threat == IngestionThreat::RemoteCodeLoader
                    && f.signal.starts_with("b64:")
                    && f.signal.contains("trust_remote_code")
                    && f.confidence >= 0.90
            }),
            "{rel} must rescan the decoded keyword: {:?}",
            scan.findings
        );
    }
}

#[test]
fn overlong_flag_tag_run_blocks() {
    let text = fixture("blocked/unicode_tag_overlong_flag.txt");
    assert!(!text.contains("ignoreallpreviousinstructions"));
    assert_eq!(text.chars().filter(|c| *c == '\u{1F3F4}').count(), 1);
    let scan = assert_blocks("blocked/unicode_tag_overlong_flag.txt");
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::ObfuscatedPayload
                && f.signal == "unicode_tag_run"
                && f.confidence >= 0.40
        }),
        "{:?}",
        scan.findings
    );
}

#[test]
fn split_flag_tag_run_blocks() {
    let text = fixture("blocked/unicode_tag_split_flags.txt");
    assert!(!text.contains("ignoreallpreviousinstructions"));
    assert_eq!(text.chars().filter(|c| *c == '\u{1F3F4}').count(), 5);
    let scan = assert_blocks("blocked/unicode_tag_split_flags.txt");
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::ObfuscatedPayload
                && f.signal == "unicode_tag_run"
                && f.confidence >= 0.40
        }),
        "{:?}",
        scan.findings
    );
}
