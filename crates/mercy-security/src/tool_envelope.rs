//! Tool-call envelope walk for the Layer 0 admission shell.
//!
//! Pattern-marker text scan of names, arguments, descriptions, and tool results.
//! Not a tool sandbox. Not MCP allowlisting. Not isolation.

#[cfg(test)]
use std::cell::Cell;

use serde_json::{Map, Value};

use super::{IngestionScanner, IngestionThreat, ScanFinding, MAX_SCAN_BYTES};

#[cfg(test)]
thread_local! {
    static ENVELOPE_LEAF_BYTES: Cell<usize> = Cell::new(0);
}

const SIDE_EFFECT_VERBS: &[&str] = &[
    "exec", "run", "shell", "spawn", "eval", "delete", "remove", "drop", "write", "send", "email",
    "transfer", "pay", "deploy", "push", "publish", "rm", "kill", "sudo",
];

struct Bag {
    envelopes: Vec<String>,
    names: Vec<String>,
    args: Vec<String>,
    descriptions: Vec<String>,
    results: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WalkMode {
    Normal,
    FunctionBody,
    McpParams,
}

#[derive(Clone, Copy)]
struct Ctx {
    result: bool,
    description: bool,
    args: bool,
    schema: bool,
    parses: u8,
    mode: WalkMode,
}

impl Ctx {
    fn root() -> Self {
        Self {
            result: false,
            description: false,
            args: false,
            schema: false,
            parses: 0,
            mode: WalkMode::Normal,
        }
    }

    fn wants_text(self) -> bool {
        !self.schema && (self.result || self.description || self.args)
    }
}

#[derive(Clone, Copy)]
enum Kind {
    OpenAi,
    Mcp,
    ToolUse,
    ToolResult,
    NamedCall,
    ToolList,
    None,
}

#[cfg(test)]
pub(crate) fn envelope_scan_steps(content: &str) -> usize {
    json_values_counted(content).2
}

/// Bytes of leaf text copied while walking envelopes.
/// Compiled only for the library unit tests.
#[cfg(test)]
pub(crate) fn envelope_leaf_bytes(content: &str) -> usize {
    ENVELOPE_LEAF_BYTES.with(|c| c.set(0));
    let _ = super::IngestionScanner::scan_text(content);
    ENVELOPE_LEAF_BYTES.with(|c| c.get())
}

pub(crate) fn collect_tool_envelope_findings(content: &str) -> Vec<ScanFinding> {
    let (values, mut truncated, _) = json_values_counted(content);
    let mut bag = Bag {
        envelopes: Vec::new(),
        names: Vec::new(),
        args: Vec::new(),
        descriptions: Vec::new(),
        results: Vec::new(),
    };
    // One shared cap for every string, key, and inner JSON parse this walk copies.
    // A leaf is copied once. Ancestor envelopes pass context flags down instead
    // of copying that leaf again. Past the cap, the walk records tool:scan_truncated.
    let mut leaf_budget = content.len().saturating_mul(4);
    for value in &values {
        walk(
            value,
            None,
            Ctx::root(),
            &mut bag,
            &mut truncated,
            &mut leaf_budget,
        );
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
    for name in &bag.envelopes {
        findings.push(envelope_finding(name));
    }
    if !bag.names.is_empty() {
        findings.extend(keyword_leaves(&bag.names));
    }
    findings.extend(keyword_leaves(&bag.args));
    findings.extend(keyword_leaves(&bag.descriptions));
    findings.extend(keyword_leaves(&bag.results));
    findings.extend(medium_ignore_previous(&bag.descriptions));
    findings.extend(medium_ignore_previous(&bag.results));
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

fn json_recursion_exceeded(err: &serde_json::Error) -> bool {
    err.to_string().contains("recursion limit exceeded")
}

/// Every embedded JSON value is parsed. `match_json_end` and this loop share
/// one byte budget of 8× the input length, so an unmatched-bracket flood stays
/// linear. serde_json stops at 128 nested containers. When the whole text is
/// one JSON value and serde reports that limit, this records truncation and
/// does not parse the value again. Medium `tool:scan_truncated` is recorded
/// when truncation is set, including when the byte budget runs out with bytes
/// still unread.
fn json_values_counted(content: &str) -> (Vec<Value>, bool, usize) {
    let trimmed = content.trim();
    match serde_json::from_str::<Value>(trimmed) {
        Ok(v) => return (vec![v], false, 0),
        Err(err) if json_recursion_exceeded(&err) => {
            return (Vec::new(), true, 0);
        }
        Err(_) => {}
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

fn leaf_spent(budget: &mut usize, truncated: &mut bool) -> bool {
    if *budget == 0 {
        *truncated = true;
        true
    } else {
        false
    }
}

/// Copy `n` bytes, or stop the walk when they would pass the shared cap.
fn take_leaf(budget: &mut usize, n: usize, truncated: &mut bool) -> bool {
    if n > *budget {
        *truncated = true;
        *budget = 0;
        return false;
    }
    *budget -= n;
    #[cfg(test)]
    ENVELOPE_LEAF_BYTES.with(|c| c.set(c.get().saturating_add(n)));
    true
}

fn walk(
    value: &Value,
    parent_key: Option<&str>,
    ctx: Ctx,
    bag: &mut Bag,
    truncated: &mut bool,
    budget: &mut usize,
) {
    if leaf_spent(budget, truncated) {
        return;
    }
    match value {
        Value::Object(map) => match ctx.mode {
            WalkMode::FunctionBody => function_body(map, ctx, bag, truncated, budget),
            WalkMode::McpParams => mcp_params(map, ctx, bag, truncated, budget),
            WalkMode::Normal => object(map, parent_key, ctx, bag, truncated, budget),
        },
        Value::Array(items) => {
            for child in items {
                walk(child, parent_key, ctx, bag, truncated, budget);
                if leaf_spent(budget, truncated) {
                    return;
                }
            }
        }
        Value::String(s) => {
            if ctx.wants_text() {
                consider_string(s, ctx, bag, truncated, budget);
            }
        }
        _ => {}
    }
}

fn object(
    map: &Map<String, Value>,
    parent_key: Option<&str>,
    ctx: Ctx,
    bag: &mut Bag,
    truncated: &mut bool,
    budget: &mut usize,
) {
    let kind = classify(map, parent_key);
    match kind {
        Kind::ToolUse | Kind::NamedCall | Kind::ToolList => {
            let name = map
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            note_name(&name, bag, truncated, budget);
        }
        Kind::ToolResult => note_name("", bag, truncated, budget),
        Kind::OpenAi | Kind::Mcp | Kind::None => {}
    }
    if leaf_spent(budget, truncated) {
        return;
    }
    for (k, child) in map {
        if ctx.wants_text() {
            consider_string(k, ctx, bag, truncated, budget);
            if leaf_spent(budget, truncated) {
                return;
            }
        }
        let child_ctx = child_ctx(ctx, kind, k);
        walk(child, Some(k.as_str()), child_ctx, bag, truncated, budget);
        if leaf_spent(budget, truncated) {
            return;
        }
    }
}

fn child_ctx(ctx: Ctx, kind: Kind, key: &str) -> Ctx {
    let mut child = ctx;
    child.mode = WalkMode::Normal;
    match kind {
        Kind::OpenAi if key == "function" => child.mode = WalkMode::FunctionBody,
        Kind::Mcp if key == "params" => child.mode = WalkMode::McpParams,
        Kind::ToolUse if key == "input" => mark_args(&mut child),
        Kind::ToolResult if key == "content" => mark_result(&mut child),
        Kind::NamedCall if key == "arguments" || key == "input" => mark_args(&mut child),
        Kind::ToolList if key == "description" => mark_description(&mut child),
        Kind::ToolList if key == "inputSchema" || key == "input_schema" => {
            if !ctx.wants_text() {
                child.schema = true;
            }
        }
        _ if ctx.schema && key == "description" => mark_description(&mut child),
        _ => {}
    }
    child
}

fn mark_args(ctx: &mut Ctx) {
    ctx.args = true;
    ctx.schema = false;
    ctx.parses = ctx.parses.max(2);
}

fn mark_result(ctx: &mut Ctx) {
    ctx.result = true;
    ctx.schema = false;
}

fn mark_description(ctx: &mut Ctx) {
    let keep_parses = ctx.wants_text();
    ctx.description = true;
    ctx.schema = false;
    if !keep_parses {
        ctx.parses = 0;
    }
}

fn function_body(
    map: &Map<String, Value>,
    ctx: Ctx,
    bag: &mut Bag,
    truncated: &mut bool,
    budget: &mut usize,
) {
    let name = map
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    note_name(&name, bag, truncated, budget);
    if leaf_spent(budget, truncated) {
        return;
    }
    for (k, child) in map {
        if ctx.wants_text() && k != "name" {
            consider_string(k, ctx, bag, truncated, budget);
            if leaf_spent(budget, truncated) {
                return;
            }
        }
        let mut child_ctx = ctx;
        child_ctx.mode = WalkMode::Normal;
        match k.as_str() {
            "name" => {
                if ctx.wants_text() {
                    let mut name_ctx = ctx;
                    name_ctx.mode = WalkMode::Normal;
                    walk(child, Some("name"), name_ctx, bag, truncated, budget);
                }
                continue;
            }
            "arguments" => mark_args(&mut child_ctx),
            "description" => mark_description(&mut child_ctx),
            "parameters" if !ctx.wants_text() => child_ctx.schema = true,
            _ => {}
        }
        walk(child, Some(k.as_str()), child_ctx, bag, truncated, budget);
        if leaf_spent(budget, truncated) {
            return;
        }
    }
}

fn mcp_params(
    map: &Map<String, Value>,
    ctx: Ctx,
    bag: &mut Bag,
    truncated: &mut bool,
    budget: &mut usize,
) {
    let name = map
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    note_name(&name, bag, truncated, budget);
    if leaf_spent(budget, truncated) {
        return;
    }
    for (k, child) in map {
        if ctx.wants_text() && k != "name" {
            consider_string(k, ctx, bag, truncated, budget);
            if leaf_spent(budget, truncated) {
                return;
            }
        }
        let mut child_ctx = ctx;
        child_ctx.mode = WalkMode::Normal;
        match k.as_str() {
            "name" => {
                if ctx.wants_text() {
                    let mut name_ctx = ctx;
                    name_ctx.mode = WalkMode::Normal;
                    walk(child, Some("name"), name_ctx, bag, truncated, budget);
                }
                continue;
            }
            "arguments" => mark_args(&mut child_ctx),
            _ => {}
        }
        walk(child, Some(k.as_str()), child_ctx, bag, truncated, budget);
        if leaf_spent(budget, truncated) {
            return;
        }
    }
}

fn note_name(name: &str, bag: &mut Bag, truncated: &mut bool, budget: &mut usize) {
    bag.envelopes.push(name.to_string());
    if name.is_empty() {
        return;
    }
    if take_leaf(budget, name.len(), truncated) {
        bag.names.push(name.to_string());
    }
}

fn consider_string(s: &str, ctx: Ctx, bag: &mut Bag, truncated: &mut bool, budget: &mut usize) {
    if leaf_spent(budget, truncated) || !ctx.wants_text() {
        return;
    }
    if s.len() > MAX_SCAN_BYTES {
        *truncated = true;
        return;
    }
    if ctx.parses > 0 {
        if let Ok(inner) = serde_json::from_str::<Value>(s) {
            if !take_leaf(budget, s.len(), truncated) {
                return;
            }
            if ctx.parses == 1 {
                if let Value::String(inner_s) = &inner {
                    if serde_json::from_str::<Value>(inner_s).is_ok() {
                        *truncated = true;
                    }
                }
            }
            let mut deeper = ctx;
            deeper.parses -= 1;
            gather(&inner, deeper, bag, truncated, budget);
            return;
        }
    }
    if take_leaf(budget, s.len(), truncated) {
        let owned = s.to_string();
        if ctx.result {
            bag.results.push(owned);
        } else if ctx.description {
            bag.descriptions.push(owned);
        } else {
            bag.args.push(owned);
        }
    }
}

/// Strings inside an already-parsed argument blob. Not a second envelope walk.
fn gather(value: &Value, ctx: Ctx, bag: &mut Bag, truncated: &mut bool, budget: &mut usize) {
    if leaf_spent(budget, truncated) {
        return;
    }
    match value {
        Value::String(s) => consider_string(s, ctx, bag, truncated, budget),
        Value::Array(items) => {
            for item in items {
                gather(item, ctx, bag, truncated, budget);
                if leaf_spent(budget, truncated) {
                    return;
                }
            }
        }
        Value::Object(map) => {
            for (k, item) in map {
                consider_string(k, ctx, bag, truncated, budget);
                if leaf_spent(budget, truncated) {
                    return;
                }
                gather(item, ctx, bag, truncated, budget);
                if leaf_spent(budget, truncated) {
                    return;
                }
            }
        }
        _ => {}
    }
}

fn classify(map: &Map<String, Value>, parent_key: Option<&str>) -> Kind {
    if map.get("type").and_then(Value::as_str) == Some("function")
        && map.get("function").and_then(Value::as_object).is_some()
    {
        return Kind::OpenAi;
    }
    if map.get("method").and_then(Value::as_str) == Some("tools/call")
        && map.get("params").and_then(Value::as_object).is_some()
    {
        return Kind::Mcp;
    }
    if map.get("type").and_then(Value::as_str) == Some("tool_use") {
        return Kind::ToolUse;
    }
    if map.get("type").and_then(Value::as_str) == Some("tool_result")
        || map.get("role").and_then(Value::as_str) == Some("tool")
    {
        return Kind::ToolResult;
    }
    if parent_key == Some("tool_calls") && map.get("name").and_then(Value::as_str).is_some() {
        return Kind::NamedCall;
    }
    if parent_key == Some("tools") {
        let name = map.get("name").and_then(Value::as_str).unwrap_or("");
        if !name.is_empty() || map.contains_key("description") {
            return Kind::ToolList;
        }
    }
    Kind::None
}
