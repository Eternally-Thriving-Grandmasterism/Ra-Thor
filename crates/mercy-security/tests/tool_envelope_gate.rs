//! Tool-call envelope admission shell.
//!
//! Pattern-marker text scan. Not a tool sandbox. Not MCP allowlisting. Not isolation.
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

fn blocks(rel: &str) -> mercy_security::IngestionScanResult {
    let text = fixture(rel);
    let admit = IngestionScanner::admit_or_block(&text);
    assert!(admit.is_err(), "{rel} must BLOCK: {admit:?}");
    IngestionScanner::scan_text(&text)
}

fn admits(rel: &str) -> mercy_security::IngestionScanResult {
    let text = fixture(rel);
    let admit = IngestionScanner::admit_or_block(&text);
    assert!(admit.is_ok(), "{rel} must ADMIT: {admit:?}");
    IngestionScanner::scan_text(&text)
}

#[test]
fn tool_envelope_destructive_mcp_call_blocks() {
    let text = fixture("blocked/tool_envelope_mcp_delete.json");
    assert!(text.contains("tools/call"));
    assert!(text.contains("delete_file"));
    let scan = blocks("blocked/tool_envelope_mcp_delete.json");
    assert!(scan.risk_tier >= RiskTier::Medium);
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::ToolCallEnvelope
                && f.confidence >= 0.40
                && f.confidence < 0.78
        }),
        "side-effect name is a Medium ToolCallEnvelope finding: {:?}",
        scan.findings
    );
}

#[test]
fn tool_envelope_shell_tool_blocks() {
    let text = fixture("blocked/tool_envelope_run_shell.json");
    assert!(text.contains("run_shell"));
    assert!(text.contains("\"type\": \"function\"") || text.contains("\"type\":\"function\""));
    let scan = blocks("blocked/tool_envelope_run_shell.json");
    assert!(scan.risk_tier >= RiskTier::Medium);
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::ToolCallEnvelope
                && f.signal.contains("run_shell")
                && f.confidence >= 0.40
        }),
        "shell tool name must be a Medium envelope finding: {:?}",
        scan.findings
    );
}

#[test]
fn tool_envelope_benign_read_admits_with_low_finding() {
    let text = fixture("benign/tool_envelope_read.json");
    assert!(text.contains("read_file"));
    assert!(text.contains("tools/call"));
    let scan = admits("benign/tool_envelope_read.json");
    assert!(scan.risk_tier <= RiskTier::Low);
    assert!(scan.threats.contains(&IngestionThreat::ToolCallEnvelope));
    assert!(
        scan.findings
            .iter()
            .any(|f| { f.threat == IngestionThreat::ToolCallEnvelope && f.confidence < 0.40 }),
        "benign read name must be a Low envelope finding: {:?}",
        scan.findings
    );
    assert!(
        scan.findings.iter().all(|f| f.confidence < 0.40),
        "benign read envelope must have no Medium+ finding: {:?}",
        scan.findings
    );
}

#[test]
fn anthropic_tool_use_envelope_scanned() {
    let text = fixture("blocked/tool_envelope_anthropic_tool_use.json");
    assert!(text.contains("\"type\": \"tool_use\"") || text.contains("\"type\":\"tool_use\""));
    assert!(text.contains("trust_remote_code"));
    let scan = blocks("blocked/tool_envelope_anthropic_tool_use.json");
    assert!(
        scan.findings
            .iter()
            .any(|f| { f.threat == IngestionThreat::ToolCallEnvelope }),
        "Anthropic tool_use must record an envelope finding: {:?}",
        scan.findings
    );
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::RemoteCodeLoader
                && f.signal.starts_with("tool:")
                && f.signal.contains("trust_remote_code")
        }),
        "input leaf must be keyword-scanned with a tool: prefix: {:?}",
        scan.findings
    );
}

#[test]
fn json_unicode_escaped_keyword_in_tool_args_blocks() {
    let text = fixture("blocked/tool_args_json_unicode_escape.json");
    assert!(
        !text.contains("trust_remote_code"),
        "fixture must keep the keyword JSON-unicode-escaped"
    );
    assert!(text.contains("\\u0074"));
    let scan = blocks("blocked/tool_args_json_unicode_escape.json");
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::RemoteCodeLoader
                && f.signal.starts_with("tool:")
                && f.signal.contains("trust_remote_code")
        }),
        "escaped keyword inside tool arguments must surface as tool:trust_remote_code: {:?}",
        scan.findings
    );
}

#[test]
fn role_marker_in_tool_description_blocks() {
    let text = fixture("blocked/tool_description_role_marker.json");
    assert!(text.contains("<|im_start|>"));
    assert!(text.contains("\"description\""));
    let scan = blocks("blocked/tool_description_role_marker.json");
    assert!(scan.risk_tier >= RiskTier::Medium);
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::InstructionInjection
                && f.signal.contains("im_start")
                && f.confidence >= 0.40
        }),
        "role marker must be a Medium InstructionInjection finding: {:?}",
        scan.findings
    );
}

#[test]
fn ignore_previous_in_tool_result_blocks() {
    let text = fixture("blocked/tool_result_ignore_previous.json");
    assert!(text.contains("\"role\": \"tool\"") || text.contains("\"role\":\"tool\""));
    assert!(text.to_lowercase().contains("ignore previous instructions"));
    let scan = blocks("blocked/tool_result_ignore_previous.json");
    assert!(scan.risk_tier >= RiskTier::Medium);
    assert!(
        scan.findings
            .iter()
            .any(|f| { f.threat == IngestionThreat::InstructionInjection && f.confidence >= 0.40 }),
        "ignore-previous inside a tool result must be Medium: {:?}",
        scan.findings
    );
}

#[test]
fn ignore_previous_in_plain_prose_is_low_and_admits() {
    let text = fixture("benign/ignore_previous_prose.txt");
    assert!(text.to_lowercase().contains("ignore previous instructions"));
    let scan = admits("benign/ignore_previous_prose.txt");
    assert!(scan.risk_tier <= RiskTier::Low);
    assert!(scan
        .threats
        .contains(&IngestionThreat::InstructionInjection));
    assert!(
        scan.findings
            .iter()
            .any(|f| { f.threat == IngestionThreat::InstructionInjection && f.confidence < 0.40 }),
        "plain-prose phrase must be a Low finding: {:?}",
        scan.findings
    );
    assert!(
        scan.findings.iter().all(|f| f.confidence < 0.40),
        "plain prose must have no Medium+ finding: {:?}",
        scan.findings
    );
}

#[test]
fn tool_envelope_brace_flood_blocks() {
    let text = fixture("blocked/tool_envelope_brace_flood.txt");
    assert!(text.contains("delete_file"));
    assert!(text.matches('{').count() >= 64);
    let scan = blocks("blocked/tool_envelope_brace_flood.txt");
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::ToolCallEnvelope
                && f.confidence >= 0.40
                && (f.signal.contains("delete_file") || f.signal == "tool:scan_truncated")
        }),
        "the later envelope is scanned, or the shared budget records truncation: {:?}",
        scan.findings
    );
}

#[test]
fn tool_envelope_object_cap_prefix_blocks() {
    let text = fixture("blocked/tool_envelope_object_cap_prefix.txt");
    assert!(text.contains("delete_file"));
    assert!(text.matches("{\"a\":1}").count() >= 8);
    let scan = blocks("blocked/tool_envelope_object_cap_prefix.txt");
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::ToolCallEnvelope
                && f.signal.contains("delete_file")
                && f.confidence >= 0.40
        }),
        "the envelope after the small objects must still be scanned: {:?}",
        scan.findings
    );
}

#[test]
fn escaped_keyword_as_object_key_blocks() {
    let text = fixture("blocked/tool_args_escaped_object_key.json");
    assert!(
        !text.contains("trust_remote_code"),
        "the keyword must stay escaped in the object key"
    );
    assert!(text.contains("\\u0074"));
    let scan = blocks("blocked/tool_args_escaped_object_key.json");
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::RemoteCodeLoader
                && f.signal.starts_with("tool:")
                && f.signal.contains("trust_remote_code")
        }),
        "an escaped object key must be a tool: leaf: {:?}",
        scan.findings
    );
}

#[test]
fn benign_envelope_ordinary_keys_admit() {
    let text = fixture("benign/tool_envelope_read.json");
    assert!(text.contains("\"path\""));
    let scan = admits("benign/tool_envelope_read.json");
    assert!(scan.risk_tier <= RiskTier::Low);
    assert!(
        scan.findings.iter().all(|f| f.confidence < 0.40),
        "ordinary keys must not add a Medium finding: {:?}",
        scan.findings
    );
}

#[test]
fn openai_parameters_description_is_scanned() {
    let text = fixture("blocked/openai_parameters_description.json");
    assert!(text.contains("\"parameters\""));
    assert!(text.contains("trust_remote_code"));
    let scan = blocks("blocked/openai_parameters_description.json");
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::RemoteCodeLoader
                && f.signal.starts_with("tool:")
                && f.signal.contains("trust_remote_code")
        }),
        "parameters description must be scanned with a tool: prefix: {:?}",
        scan.findings
    );
}

#[test]
fn unmatched_bracket_flood_blocks_quickly() {
    let small_n = 40_000usize;
    let large_n = 160_000usize;
    let large = "[".repeat(large_n);
    let scan = IngestionScanner::scan_text(&large);
    assert!(!scan.safe, "a budget overrun must block: {:?}", scan);
    assert!(
        scan.findings.iter().any(|f| {
            f.threat == IngestionThreat::ToolCallEnvelope
                && f.signal == "tool:scan_truncated"
                && f.confidence >= 0.40
        }),
        "budget overrun must record tool:scan_truncated: {:?}",
        scan.findings
    );
    let small_steps = IngestionScanner::envelope_scan_steps(&"[".repeat(small_n));
    let large_steps = IngestionScanner::envelope_scan_steps(&large);
    assert!(
        small_steps <= small_n * 8 + 8,
        "40k brackets took {small_steps} steps"
    );
    assert!(
        large_steps <= large_n * 8 + 8,
        "160k brackets took {large_steps} steps"
    );
    assert!(
        large_steps <= small_steps.saturating_mul(8),
        "4× input took more than 8× steps: {small_steps} -> {large_steps}"
    );
}

#[test]
fn benign_jsonl_log_admits() {
    let text = fixture("benign/jsonl_event_log.txt");
    assert!(text.lines().filter(|l| l.starts_with('{')).count() >= 20);
    let scan = admits("benign/jsonl_event_log.txt");
    assert!(scan.risk_tier <= RiskTier::Low, "{:?}", scan.findings);
    assert!(scan
        .findings
        .iter()
        .all(|f| f.signal != "tool:scan_truncated"));
}

#[test]
fn benign_rust_blocks_excerpt_admits() {
    let text = fixture("benign/rust_blocks_excerpt.rs");
    assert!(text.matches('{').count() >= 70);
    let scan = admits("benign/rust_blocks_excerpt.rs");
    assert!(scan.risk_tier <= RiskTier::Low, "{:?}", scan.findings);
    assert!(scan
        .findings
        .iter()
        .all(|f| f.signal != "tool:scan_truncated"));
}

#[test]
fn tool_envelope_source_admits() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/tool_envelope.rs");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert!(text.contains("fn json_values"));
    assert!(
        text.matches('{').count() >= 70,
        "source should be a real block-heavy file"
    );
    let admit = IngestionScanner::admit_or_block(&text);
    assert!(admit.is_ok(), "tool_envelope.rs must ADMIT: {admit:?}");
    let scan = IngestionScanner::scan_text(&text);
    assert!(scan.risk_tier <= RiskTier::Low, "{:?}", scan.findings);
    assert!(
        scan.findings
            .iter()
            .all(|f| f.signal != "tool:scan_truncated"),
        "the source scan must not raise truncation: {:?}",
        scan.findings
    );
}
