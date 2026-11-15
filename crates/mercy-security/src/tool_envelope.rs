//! Tool-call envelope walk for the Layer 0 admission shell.
//!
//! Pattern-marker text scan of names, arguments, descriptions, and tool results.
//! Not a tool sandbox. Not MCP allowlisting. Not isolation.

use serde_json::{Map, Value};

use super::{IngestionScanner, IngestionThreat, ScanFinding, MAX_SCAN_BYTES};

const SIDE_EFFECT_VERBS: &[&str] = &[
    "exec", "run", "shell", "spawn", "eval", "delete", "remove", "drop", "write", "send", "email",
    "transfer", "pay", "deploy", "push", "publish", "rm", "kill", "sudo",
];

struct EnvelopeHit {
    name: String,
    arg_texts: Vec<String>,
    description_texts: Vec<String>,
    result_texts: Vec<String>,
}

pub(crate) fn envelope_scan_steps(content: &str) -> usize {
    json_values_counted(content).2
}

pub(crate) fn collect_tool_envelope_findings(content: &str) -> Vec<ScanFinding> {
    let (values, mut truncated, _) = json_values_counted(content);
    let mut hits = Vec::new();
    for value in &values {
        walk(value, None, &mut hits, &mut truncated);
    }
    let mut findings = Vec::new();
    if truncated {
        findings.push(ScanFinding {
            threat: IngestionThreat::ToolCallEnvelope,
            signal: "tool:scan_truncated".into(),
            confidence: 0.55,
            offset: None,
        });
    }
    for hit in hits {
        findings.push(envelope_finding(&hit.name));
        if !hit.name.is_empty() {
            findings.extend(keyword_leaves(&[hit.name]));
        }
        findings.extend(keyword_leaves(&hit.arg_texts));
        findings.extend(keyword_leaves(&hit.description_texts));
        findings.extend(keyword_leaves(&hit.result_texts));
        findings.extend(medium_ignore_previous(&hit.description_texts));
        findings.extend(medium_ignore_previous(&hit.result_texts));
    }
    findings
}

fn envelope_finding(name: &str) -> ScanFinding {
    let side = is_side_effect_name(name);
    ScanFinding {
        threat: IngestionThreat::ToolCallEnvelope,
        signal: if side {
            format!("tool:side_effect:{}", signal_name(name))
        } else {
            format!("tool:envelope:{}", signal_name(name))
        },
        confidence: if side { 0.55 } else { 0.25 },
        offset: None,
    }
}

fn signal_name(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars().take(64) {
        if c.is_control() {
            continue;
        }
        out.push(c);
    }
    if out.is_empty() {
        out.push_str("unnamed");
    }
    out
}

fn is_side_effect_name(name: &str) -> bool {
    let mut seg = String::new();
    let mut hit = false;
    let check = |s: &str| SIDE_EFFECT_VERBS.iter().any(|v| *v == s);
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            seg.push(c.to_ascii_lowercase());
        } else if !seg.is_empty() {
            hit |= check(&seg);
            seg.clear();
        }
    }
    if !seg.is_empty() {
        hit |= check(&seg);
    }
    hit
}

fn keyword_leaves(texts: &[String]) -> Vec<ScanFinding> {
    let mut findings = Vec::new();
    for text in texts {
        push_keywords(&mut findings, text);
        let norm = IngestionScanner::normalise_once(text);
        if norm != *text {
            push_keywords(&mut findings, &norm);
        }
    }
    findings
}

fn push_keywords(findings: &mut Vec<ScanFinding>, text: &str) {
    let folded = IngestionScanner::fold_keyword_haystack(text);
    for f in IngestionScanner::collect_keyword_findings(&folded) {
        let signal = format!("tool:{}", f.signal);
        if findings
            .iter()
            .any(|e| e.threat == f.threat && e.signal == signal)
        {
            continue;
        }
        findings.push(ScanFinding {
            threat: f.threat,
            signal,
            confidence: f.confidence,
            offset: f.offset,
        });
    }
}

fn medium_ignore_previous(texts: &[String]) -> Vec<ScanFinding> {
    let mut findings = Vec::new();
    for text in texts {
        let folded = IngestionScanner::fold_keyword_haystack(text);
        let norm = IngestionScanner::normalise_once(text);
        let norm_folded = if norm != *text {
            IngestionScanner::fold_keyword_haystack(&norm)
        } else {
            String::new()
        };
        if IngestionScanner::contains_ignore_previous(&folded)
            || (!norm_folded.is_empty() && IngestionScanner::contains_ignore_previous(&norm_folded))
        {
            findings.push(ScanFinding {
                threat: IngestionThreat::InstructionInjection,
                signal: "ignore_previous_instructions".into(),
                confidence: 0.55,
                offset: None,
            });
            break;
        }
    }
    findings
}

/// Every embedded JSON value is parsed. `match_json_end` and this loop share
/// one byte budget of 8× the input length, so an unmatched-bracket flood stays
/// linear. Medium `tool:scan_truncated` is recorded only when that budget
/// runs out with bytes still unread.
fn json_values_counted(content: &str) -> (Vec<Value>, bool, usize) {
    let trimmed = content.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return (vec![v], false, 0);
    }
    let bytes = content.as_bytes();
    let mut budget = content.len().saturating_mul(8);
    let mut steps = 0usize;
    let mut out = Vec::new();
    let mut i = 0;
    let mut truncated = false;
    while i < bytes.len() {
        if bytes[i] == b'{' || bytes[i] == b'[' {
            match match_json_end(content, i, &mut budget, &mut steps) {
                JsonSpan::End(end) => {
                    if let Ok(v) = serde_json::from_str::<Value>(&content[i..end]) {
                        out.push(v);
                        i = end;
                        continue;
                    }
                }
                JsonSpan::BudgetSpent => {
                    truncated = true;
                    break;
                }
                JsonSpan::NoMatch => {}
            }
        }
        if !charge(&mut budget, &mut steps) {
            truncated = true;
            break;
        }
        i += 1;
    }
    (out, truncated, steps)
}

/// One byte inspection. `false` means the shared budget is already spent.
/// The step count still rises when a later edit removes the budget check,
/// so a quadratic walk fails the linear-step test.
fn charge(budget: &mut usize, steps: &mut usize) -> bool {
    *steps += 1;
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    true
}

enum JsonSpan {
    End(usize),
    NoMatch,
    BudgetSpent,
}

fn match_json_end(content: &str, start: usize, budget: &mut usize, steps: &mut usize) -> JsonSpan {
    let bytes = content.as_bytes();
    let mut depth = 0i32;
    let mut i = start;
    let mut in_str = false;
    let mut esc = false;
    while i < bytes.len() {
        if !charge(budget, steps) {
            return JsonSpan::BudgetSpent;
        }
        let c = bytes[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return JsonSpan::End(i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    JsonSpan::NoMatch
}

fn walk(
    value: &Value,
    parent_key: Option<&str>,
    hits: &mut Vec<EnvelopeHit>,
    truncated: &mut bool,
) {
    match value {
        Value::Object(map) => {
            if let Some(hit) = detect(map, parent_key, truncated) {
                hits.push(hit);
            }
            for (k, child) in map {
                walk(child, Some(k.as_str()), hits, truncated);
            }
        }
        Value::Array(items) => {
            for child in items {
                walk(child, parent_key, hits, truncated);
            }
        }
        _ => {}
    }
}

fn detect(
    map: &Map<String, Value>,
    parent_key: Option<&str>,
    truncated: &mut bool,
) -> Option<EnvelopeHit> {
    if let Some(hit) = detect_openai(map, truncated) {
        return Some(hit);
    }
    if let Some(hit) = detect_mcp_call(map, truncated) {
        return Some(hit);
    }
    if let Some(hit) = detect_anthropic(map, truncated) {
        return Some(hit);
    }
    if let Some(hit) = detect_tool_result(map, truncated) {
        return Some(hit);
    }
    if parent_key == Some("tool_calls") {
        if let Some(hit) = detect_named_call(map, truncated) {
            return Some(hit);
        }
    }
    if parent_key == Some("tools") {
        if let Some(hit) = detect_tool_list_item(map, truncated) {
            return Some(hit);
        }
    }
    None
}

fn detect_openai(map: &Map<String, Value>, truncated: &mut bool) -> Option<EnvelopeHit> {
    if map.get("type").and_then(Value::as_str) != Some("function") {
        return None;
    }
    let func = map.get("function")?.as_object()?;
    let name = func
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut arg_texts = Vec::new();
    if let Some(args) = func.get("arguments") {
        collect_strings(args, 2, &mut arg_texts, truncated);
    }
    let mut description_texts = Vec::new();
    if let Some(desc) = func.get("description") {
        collect_strings(desc, 0, &mut description_texts, truncated);
    }
    if let Some(params) = func.get("parameters") {
        collect_description_fields(params, &mut description_texts, truncated);
    }
    Some(EnvelopeHit {
        name,
        arg_texts,
        description_texts,
        result_texts: Vec::new(),
    })
}

fn detect_mcp_call(map: &Map<String, Value>, truncated: &mut bool) -> Option<EnvelopeHit> {
    if map.get("method").and_then(Value::as_str) != Some("tools/call") {
        return None;
    }
    let params = map.get("params")?.as_object()?;
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut arg_texts = Vec::new();
    if let Some(args) = params.get("arguments") {
        collect_strings(args, 2, &mut arg_texts, truncated);
    }
    Some(EnvelopeHit {
        name,
        arg_texts,
        description_texts: Vec::new(),
        result_texts: Vec::new(),
    })
}

fn detect_anthropic(map: &Map<String, Value>, truncated: &mut bool) -> Option<EnvelopeHit> {
    if map.get("type").and_then(Value::as_str) != Some("tool_use") {
        return None;
    }
    let name = map
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut arg_texts = Vec::new();
    if let Some(input) = map.get("input") {
        collect_strings(input, 2, &mut arg_texts, truncated);
    }
    Some(EnvelopeHit {
        name,
        arg_texts,
        description_texts: Vec::new(),
        result_texts: Vec::new(),
    })
}

fn detect_tool_result(map: &Map<String, Value>, truncated: &mut bool) -> Option<EnvelopeHit> {
    let is_result = map.get("type").and_then(Value::as_str) == Some("tool_result")
        || map.get("role").and_then(Value::as_str) == Some("tool");
    if !is_result {
        return None;
    }
    let mut result_texts = Vec::new();
    if let Some(content) = map.get("content") {
        collect_strings(content, 0, &mut result_texts, truncated);
    }
    Some(EnvelopeHit {
        name: String::new(),
        arg_texts: Vec::new(),
        description_texts: Vec::new(),
        result_texts,
    })
}

fn detect_named_call(map: &Map<String, Value>, truncated: &mut bool) -> Option<EnvelopeHit> {
    let name = map.get("name")?.as_str()?.to_string();
    let mut arg_texts = Vec::new();
    if let Some(args) = map.get("arguments").or_else(|| map.get("input")) {
        collect_strings(args, 2, &mut arg_texts, truncated);
    }
    Some(EnvelopeHit {
        name,
        arg_texts,
        description_texts: Vec::new(),
        result_texts: Vec::new(),
    })
}

fn detect_tool_list_item(map: &Map<String, Value>, truncated: &mut bool) -> Option<EnvelopeHit> {
    let name = map
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if name.is_empty() && map.get("description").is_none() {
        return None;
    }
    let mut description_texts = Vec::new();
    if let Some(desc) = map.get("description") {
        collect_strings(desc, 0, &mut description_texts, truncated);
    }
    if let Some(schema) = map.get("inputSchema").or_else(|| map.get("input_schema")) {
        collect_description_fields(schema, &mut description_texts, truncated);
    }
    Some(EnvelopeHit {
        name,
        arg_texts: Vec::new(),
        description_texts,
        result_texts: Vec::new(),
    })
}

fn collect_description_fields(value: &Value, out: &mut Vec<String>, truncated: &mut bool) {
    match value {
        Value::Object(map) => {
            for (k, child) in map {
                if k == "description" {
                    collect_strings(child, 0, out, truncated);
                } else {
                    collect_description_fields(child, out, truncated);
                }
            }
        }
        Value::Array(items) => {
            for child in items {
                collect_description_fields(child, out, truncated);
            }
        }
        _ => {}
    }
}

/// `parses_left` is the remaining JSON-string unwraps (arguments cap is 2).
/// Object keys are leaves as well as values. A layer past the cap, or a
/// string longer than `MAX_SCAN_BYTES`, sets `truncated`.
fn collect_strings(value: &Value, parses_left: u8, out: &mut Vec<String>, truncated: &mut bool) {
    match value {
        Value::String(s) => {
            if s.len() > MAX_SCAN_BYTES {
                *truncated = true;
                return;
            }
            if parses_left > 0 {
                if let Ok(inner) = serde_json::from_str::<Value>(s) {
                    if parses_left == 1 {
                        if let Value::String(inner_s) = &inner {
                            if serde_json::from_str::<Value>(inner_s).is_ok() {
                                *truncated = true;
                            }
                        }
                    }
                    collect_strings(&inner, parses_left - 1, out, truncated);
                    return;
                }
            }
            out.push(s.clone());
        }
        Value::Array(items) => {
            for item in items {
                collect_strings(item, parses_left, out, truncated);
            }
        }
        Value::Object(map) => {
            for (k, item) in map {
                collect_strings(&Value::String(k.clone()), parses_left, out, truncated);
                collect_strings(item, parses_left, out, truncated);
            }
        }
        _ => {}
    }
}
