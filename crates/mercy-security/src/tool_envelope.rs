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

pub(crate) fn collect_tool_envelope_findings(content: &str) -> Vec<ScanFinding> {
    let mut hits = Vec::new();
    for value in json_values(content) {
        walk(&value, None, &mut hits);
    }
    let mut findings = Vec::new();
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

fn json_values(content: &str) -> Vec<Value> {
    let trimmed = content.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return vec![v];
    }
    let bytes = content.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let mut attempts = 0usize;
    while i < bytes.len() && out.len() < 8 && attempts < 64 {
        if bytes[i] == b'{' || bytes[i] == b'[' {
            attempts += 1;
            if let Some(end) = match_json_end(content, i) {
                if let Ok(v) = serde_json::from_str::<Value>(&content[i..end]) {
                    out.push(v);
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

fn match_json_end(content: &str, start: usize) -> Option<usize> {
    let bytes = content.as_bytes();
    let mut depth = 0i32;
    let mut i = start;
    let mut in_str = false;
    let mut esc = false;
    while i < bytes.len() {
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
                    return Some(i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn walk(value: &Value, parent_key: Option<&str>, hits: &mut Vec<EnvelopeHit>) {
    match value {
        Value::Object(map) => {
            if let Some(hit) = detect(map, parent_key) {
                hits.push(hit);
            }
            for (k, child) in map {
                walk(child, Some(k.as_str()), hits);
            }
        }
        Value::Array(items) => {
            for child in items {
                walk(child, parent_key, hits);
            }
        }
        _ => {}
    }
}

fn detect(map: &Map<String, Value>, parent_key: Option<&str>) -> Option<EnvelopeHit> {
    if let Some(hit) = detect_openai(map) {
        return Some(hit);
    }
    if let Some(hit) = detect_mcp_call(map) {
        return Some(hit);
    }
    if let Some(hit) = detect_anthropic(map) {
        return Some(hit);
    }
    if let Some(hit) = detect_tool_result(map) {
        return Some(hit);
    }
    if parent_key == Some("tool_calls") {
        if let Some(hit) = detect_named_call(map) {
            return Some(hit);
        }
    }
    if parent_key == Some("tools") {
        if let Some(hit) = detect_tool_list_item(map) {
            return Some(hit);
        }
    }
    None
}

fn detect_openai(map: &Map<String, Value>) -> Option<EnvelopeHit> {
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
        collect_strings(args, 2, &mut arg_texts);
    }
    let mut description_texts = Vec::new();
    if let Some(desc) = func.get("description") {
        collect_strings(desc, 0, &mut description_texts);
    }
    Some(EnvelopeHit {
        name,
        arg_texts,
        description_texts,
        result_texts: Vec::new(),
    })
}

fn detect_mcp_call(map: &Map<String, Value>) -> Option<EnvelopeHit> {
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
        collect_strings(args, 2, &mut arg_texts);
    }
    Some(EnvelopeHit {
        name,
        arg_texts,
        description_texts: Vec::new(),
        result_texts: Vec::new(),
    })
}

fn detect_anthropic(map: &Map<String, Value>) -> Option<EnvelopeHit> {
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
        collect_strings(input, 2, &mut arg_texts);
    }
    Some(EnvelopeHit {
        name,
        arg_texts,
        description_texts: Vec::new(),
        result_texts: Vec::new(),
    })
}

fn detect_tool_result(map: &Map<String, Value>) -> Option<EnvelopeHit> {
    let is_result = map.get("type").and_then(Value::as_str) == Some("tool_result")
        || map.get("role").and_then(Value::as_str) == Some("tool");
    if !is_result {
        return None;
    }
    let mut result_texts = Vec::new();
    if let Some(content) = map.get("content") {
        collect_strings(content, 0, &mut result_texts);
    }
    Some(EnvelopeHit {
        name: String::new(),
        arg_texts: Vec::new(),
        description_texts: Vec::new(),
        result_texts,
    })
}

fn detect_named_call(map: &Map<String, Value>) -> Option<EnvelopeHit> {
    let name = map.get("name")?.as_str()?.to_string();
    let mut arg_texts = Vec::new();
    if let Some(args) = map.get("arguments").or_else(|| map.get("input")) {
        collect_strings(args, 2, &mut arg_texts);
    }
    Some(EnvelopeHit {
        name,
        arg_texts,
        description_texts: Vec::new(),
        result_texts: Vec::new(),
    })
}

fn detect_tool_list_item(map: &Map<String, Value>) -> Option<EnvelopeHit> {
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
        collect_strings(desc, 0, &mut description_texts);
    }
    if let Some(schema) = map.get("inputSchema").or_else(|| map.get("input_schema")) {
        collect_description_fields(schema, &mut description_texts);
    }
    Some(EnvelopeHit {
        name,
        arg_texts: Vec::new(),
        description_texts,
        result_texts: Vec::new(),
    })
}

fn collect_description_fields(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (k, child) in map {
                if k == "description" {
                    collect_strings(child, 0, out);
                } else {
                    collect_description_fields(child, out);
                }
            }
        }
        Value::Array(items) => {
            for child in items {
                collect_description_fields(child, out);
            }
        }
        _ => {}
    }
}

/// `parses_left` is the remaining JSON-string unwraps (arguments cap is 2).
fn collect_strings(value: &Value, parses_left: u8, out: &mut Vec<String>) {
    match value {
        Value::String(s) => {
            if s.len() > MAX_SCAN_BYTES {
                return;
            }
            if parses_left > 0 {
                if let Ok(inner) = serde_json::from_str::<Value>(s) {
                    collect_strings(&inner, parses_left - 1, out);
                    return;
                }
            }
            out.push(s.clone());
        }
        Value::Array(items) => {
            for item in items {
                collect_strings(item, parses_left, out);
            }
        }
        Value::Object(map) => {
            for item in map.values() {
                collect_strings(item, parses_left, out);
            }
        }
        _ => {}
    }
}
