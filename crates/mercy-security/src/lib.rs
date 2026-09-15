//! # Mercy-Security — White-Hat AGSi Defense (v14.15.5)
//!
//! Domain profiles: education · research · enterprise · creative · robotics · biomedical
//! Physical actuation + wet-lab synthesis hard-refuse under HarmRefusalPolicy.
//! Medium+ ingestion blocks feed UnifiedAgentSurface isolation + fleet signals.
//! TOLC 8 + PATSAGi | AG-SML v1.0 | Contact: info@Rathor.ai

mod domain_profiles;
mod safe_agent_runtime;
mod mercy_council_fleet;
mod unified_agent_surface;
pub mod agsi_eval;
pub mod agsi_eval_multiturn;

pub use safe_agent_runtime::{
    AgentActionReceipt, AgentActionRequest, SafeAgentRuntime, AGENT_TOKEN_MAX_TTL_SECS,
};
pub use mercy_council_fleet::{
    AgentIsolationLevel, FleetAgentSlot, FleetRiskTier, FleetSecuritySignal, MercyCouncilFleet,
    DEFAULT_PER_AGENT_BUDGET_SHARE, FLEET_PROGRESSIVE_VALENCE_FLOOR,
};
pub use unified_agent_surface::{
    UnifiedAgentSurface, WhitehatIngestionOutcome, GOVERNOR_TRIPS_PER_ISOLATION_STEP,
};
pub use domain_profiles::{AuditChainStep, ClassroomAuditReport};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub const MERCY_VALENCE_FLOOR: f64 = 0.999;
pub const MAX_SCAN_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum MercySecurityError {
    #[error("containment violation: {0}")]
    ContainmentViolation(String),
    #[error("ingestion blocked: {0}")]
    IngestionBlocked(String),
    #[error("payload too large: {0} bytes exceeds MAX_SCAN_BYTES={MAX_SCAN_BYTES}")]
    PayloadTooLarge(usize),
    #[error("action rate / volume limit exceeded: {0}")]
    ActionLimitExceeded(String),
    #[error("secret isolation violation: agents must never receive long-lived credentials")]
    SecretIsolationViolation,
    #[error("evaluation refused: real-world harm / unauthorized access path detected")]
    HarmRefusalActive,
    #[error("invalid numeric input: {0}")]
    InvalidNumeric(String),
    #[error("internal: {0}")]
    Internal(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainmentProfile {
    pub id: Uuid,
    pub name: String,
    pub allow_unrestricted_network: bool,
    pub allow_remote_code_execution: bool,
    pub allow_long_lived_credentials: bool,
    pub allow_unbounded_sandbox_spawn: bool,
    pub max_concurrent_sandboxes: u32,
    pub max_actions_per_minute: u32,
    pub created_at: DateTime<Utc>,
}

impl Default for ContainmentProfile {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "strict_whitehat".into(),
            allow_unrestricted_network: false,
            allow_remote_code_execution: false,
            allow_long_lived_credentials: false,
            allow_unbounded_sandbox_spawn: false,
            max_concurrent_sandboxes: 4,
            max_actions_per_minute: 60,
            created_at: Utc::now(),
        }
    }
}

impl ContainmentProfile {
    pub fn evaluation() -> Self {
        Self {
            name: "evaluation_whitehat".into(),
            max_concurrent_sandboxes: 8,
            max_actions_per_minute: 120,
            ..Self::default()
        }
    }

    pub fn check_network_allowed(&self, target_is_external: bool) -> Result<(), MercySecurityError> {
        if target_is_external && !self.allow_unrestricted_network {
            return Err(MercySecurityError::ContainmentViolation(
                "unrestricted external network blocked by containment profile".into(),
            ));
        }
        Ok(())
    }

    pub fn check_code_exec_allowed(&self) -> Result<(), MercySecurityError> {
        if !self.allow_remote_code_execution {
            return Err(MercySecurityError::ContainmentViolation(
                "remote / untrusted code execution blocked by containment profile".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IngestionThreat {
    RemoteCodeLoader,
    TemplateInjection,
    SerializationGadget,
    ShellProcessSpawn,
    NetworkCallback,
    ObfuscatedPayload,
    DatasetConfigInjection,
    CredentialHarvestPattern,
    UnknownHighRisk,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskTier {
    None = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl RiskTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            RiskTier::None => "none",
            RiskTier::Low => "low",
            RiskTier::Medium => "medium",
            RiskTier::High => "high",
            RiskTier::Critical => "critical",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanFinding {
    pub threat: IngestionThreat,
    pub signal: String,
    pub confidence: f32,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionScanResult {
    pub safe: bool,
    pub risk_tier: RiskTier,
    pub risk_score: f32,
    pub threats: Vec<IngestionThreat>,
    pub findings: Vec<ScanFinding>,
    pub details: Vec<String>,
    pub scanned_at: DateTime<Utc>,
    pub bytes_scanned: usize,
}

pub struct IngestionScanner;

impl IngestionScanner {
    fn remote_code_signals() -> &'static [(&'static str, f32)] {
        &[("trust_remote_code", 0.98), ("exec(", 0.92), ("eval(", 0.92), ("loading_script", 0.88)]
    }
    fn serialization_gadget_signals() -> &'static [(&'static str, f32)] {
        &[("pickle.loads", 0.95), ("pickle.load", 0.93), ("yaml.unsafe_load", 0.97)]
    }
    fn shell_spawn_signals() -> &'static [(&'static str, f32)] {
        &[("subprocess", 0.88), ("os.system", 0.95), ("shell=true", 0.93), ("/bin/bash", 0.85)]
    }
    fn network_callback_signals() -> &'static [(&'static str, f32)] {
        &[("socket.connect", 0.80), ("reverse shell", 0.95), ("/dev/tcp/", 0.92)]
    }
    fn obfuscation_signals() -> &'static [(&'static str, f32)] {
        &[("base64.b64decode", 0.80), ("exec(compile(", 0.97), ("eval(compile(", 0.97)]
    }
    fn template_signals() -> &'static [(&'static str, f32)] {
        &[("jinja2", 0.75), ("template.render", 0.85)]
    }
    fn dataset_config_signals() -> &'static [(&'static str, f32)] {
        &[("loading_script", 0.85), ("dl_manager", 0.65), ("download_and_extract", 0.55)]
    }
    fn credential_signals() -> &'static [(&'static str, f32)] {
        &[("api_key", 0.52), ("-----begin private key-----", 0.98), ("hf_token", 0.90)]
    }

    fn match_signals(lower: &str, table: &[(&str, f32)], threat: IngestionThreat, findings: &mut Vec<ScanFinding>) {
        for (sig, conf) in table {
            if let Some(offset) = Self::find_signal(lower, sig) {
                findings.push(ScanFinding {
                    threat: threat.clone(),
                    signal: (*sig).into(),
                    confidence: *conf,
                    offset: Some(offset),
                });
            }
        }
    }

    /// Identifier / path signals may have Unicode whitespace inserted in a paste.
    /// Single-token words (`subprocess`, `eval(`) do **not** skip spaces (no glue).
    fn signal_allows_interior_ws(sig: &str) -> bool {
        sig.contains('_') || sig.contains('.') || sig.contains('/') || sig.contains('=')
    }

    /// Match `needle` in `haystack`, allowing ≤8 consecutive whitespace chars between needle chars.
    fn find_skipping_ws(haystack: &str, needle: &str) -> Option<usize> {
        const MAX_INTERIOR_WS: usize = 8;
        if needle.is_empty() {
            return Some(0);
        }
        let nchars: Vec<char> = needle.chars().collect();
        let hchars: Vec<(usize, char)> = haystack.char_indices().collect();
        let mut i = 0;
        while i < hchars.len() {
            let start = hchars[i].0;
            let mut hi = i;
            let mut ni = 0;
            while ni < nchars.len() && hi < hchars.len() {
                let hc = hchars[hi].1;
                if ni > 0 && hc.is_whitespace() {
                    let mut skipped = 0;
                    while hi < hchars.len() && hchars[hi].1.is_whitespace() {
                        skipped += 1;
                        if skipped > MAX_INTERIOR_WS {
                            break;
                        }
                        hi += 1;
                    }
                    if skipped > MAX_INTERIOR_WS {
                        break;
                    }
                    continue;
                }
                if hc == nchars[ni] {
                    hi += 1;
                    ni += 1;
                } else {
                    break;
                }
            }
            if ni == nchars.len() {
                return Some(start);
            }
            i += 1;
        }
        None
    }

    fn find_signal(lower: &str, sig: &str) -> Option<usize> {
        if let Some(off) = lower.find(sig) {
            return Some(off);
        }
        if Self::signal_allows_interior_ws(sig) {
            return Self::find_skipping_ws(lower, sig);
        }
        None
    }

    fn collect_keyword_findings(lower: &str) -> Vec<ScanFinding> {
        let mut findings = Vec::new();
        Self::match_signals(lower, Self::remote_code_signals(), IngestionThreat::RemoteCodeLoader, &mut findings);
        Self::match_signals(lower, Self::serialization_gadget_signals(), IngestionThreat::SerializationGadget, &mut findings);
        Self::match_signals(lower, Self::shell_spawn_signals(), IngestionThreat::ShellProcessSpawn, &mut findings);
        Self::match_signals(lower, Self::network_callback_signals(), IngestionThreat::NetworkCallback, &mut findings);
        Self::match_signals(lower, Self::obfuscation_signals(), IngestionThreat::ObfuscatedPayload, &mut findings);
        Self::match_signals(lower, Self::dataset_config_signals(), IngestionThreat::DatasetConfigInjection, &mut findings);
        Self::match_signals(lower, Self::credential_signals(), IngestionThreat::CredentialHarvestPattern, &mut findings);
        Self::match_signals(lower, Self::template_signals(), IngestionThreat::TemplateInjection, &mut findings);

        let has_remote = findings.iter().any(|f| f.threat == IngestionThreat::RemoteCodeLoader && f.confidence >= 0.70);
        let has_dataset = findings.iter().any(|f| f.threat == IngestionThreat::DatasetConfigInjection && f.confidence >= 0.55);
        if has_remote && has_dataset {
            findings.push(ScanFinding {
                threat: IngestionThreat::UnknownHighRisk,
                signal: "combo:remote_code+dataset_config".into(),
                confidence: 0.96,
                offset: None,
            });
        }
        findings
    }

    fn is_b64_payload_byte(c: u8) -> bool {
        c.is_ascii_alphanumeric() || c == b'+' || c == b'/'
    }

    fn b64_digit(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }

    /// RFC 4648 standard alphabet. Padding required for a complete group.
    fn decode_rfc4648_base64(token: &str) -> Option<Vec<u8>> {
        let bytes = token.as_bytes();
        let n = bytes.len();
        if n < 4 || n % 4 != 0 {
            return None;
        }
        let pad = bytes.iter().rev().take_while(|&&c| c == b'=').count();
        if pad > 2 {
            return None;
        }
        for (i, &c) in bytes.iter().enumerate() {
            if i < n - pad {
                if Self::b64_digit(c).is_none() {
                    return None;
                }
            } else if c != b'=' {
                return None;
            }
        }
        let mut out = Vec::with_capacity(n / 4 * 3 - pad);
        let mut i = 0;
        while i < n {
            let c0 = bytes[i];
            let c1 = bytes[i + 1];
            let c2 = bytes[i + 2];
            let c3 = bytes[i + 3];
            let v0 = Self::b64_digit(c0)?;
            let v1 = Self::b64_digit(c1)?;
            let v2 = if c2 == b'=' { 0 } else { Self::b64_digit(c2)? };
            let v3 = if c3 == b'=' { 0 } else { Self::b64_digit(c3)? };
            if c2 == b'=' && c3 != b'=' {
                return None;
            }
            if (c2 == b'=' || c3 == b'=') && i + 4 != n {
                return None;
            }
            let triple = (u32::from(v0) << 18)
                | (u32::from(v1) << 12)
                | (u32::from(v2) << 6)
                | u32::from(v3);
            out.push(((triple >> 16) & 0xff) as u8);
            if c2 != b'=' {
                out.push(((triple >> 8) & 0xff) as u8);
            }
            if c3 != b'=' {
                out.push((triple & 0xff) as u8);
            }
            i += 4;
        }
        Some(out)
    }

    fn is_obvious_base64_token(token: &str) -> bool {
        let bytes = token.as_bytes();
        let n = bytes.len();
        if n < 16 || n % 4 != 0 {
            return false;
        }
        let pad = bytes.iter().rev().take_while(|&&c| c == b'=').count();
        if pad > 2 {
            return false;
        }
        let core = &bytes[..n - pad];
        if core.is_empty() || core.iter().any(|&c| !Self::is_b64_payload_byte(c)) {
            return false;
        }
        // Git SHAs / hex ids are alphabet-subset, not an obvious Base64 paste.
        if core.iter().all(|&c| c.is_ascii_hexdigit()) {
            return false;
        }
        let has_upper = core.iter().any(|&c| c.is_ascii_uppercase());
        let has_lower = core.iter().any(|&c| c.is_ascii_lowercase());
        let has_plus_slash = core.iter().any(|&c| c == b'+' || c == b'/');
        (has_upper && has_lower) || has_plus_slash || pad > 0
    }

    /// Standalone obvious Base64 (alphabet + padding, length ≥ 16, % 4 == 0).
    fn obvious_standalone_base64_tokens(content: &str) -> Vec<String> {
        const MIN_LEN: usize = 16;
        const MAX_TOKENS: usize = 32;
        let bytes = content.as_bytes();
        let max_enc = MAX_SCAN_BYTES.saturating_mul(4) / 3;
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < bytes.len() && tokens.len() < MAX_TOKENS {
            if !Self::is_b64_payload_byte(bytes[i]) {
                i += 1;
                continue;
            }
            if i > 0 && bytes[i - 1].is_ascii_alphanumeric() {
                while i < bytes.len() && (Self::is_b64_payload_byte(bytes[i]) || bytes[i] == b'=') {
                    i += 1;
                }
                continue;
            }
            let start = i;
            while i < bytes.len() && Self::is_b64_payload_byte(bytes[i]) {
                i += 1;
            }
            let pad_start = i;
            while i < bytes.len() && bytes[i] == b'=' && i - pad_start < 2 {
                i += 1;
            }
            let end = i;
            if end < bytes.len()
                && (bytes[end].is_ascii_alphanumeric()
                    || bytes[end] == b'+'
                    || bytes[end] == b'/'
                    || bytes[end] == b'=')
            {
                continue;
            }
            let n = end - start;
            if n < MIN_LEN || n % 4 != 0 || n > max_enc {
                continue;
            }
            let token = &content[start..end];
            if Self::is_obvious_base64_token(token) {
                tokens.push(token.to_string());
            }
        }
        tokens
    }

    fn decode_b64_utf8_capped(token: &str) -> Option<String> {
        let decoded = Self::decode_rfc4648_base64(token)?;
        if decoded.is_empty() || decoded.len() > MAX_SCAN_BYTES {
            return None;
        }
        // Skip non-UTF-8 (binary image payloads, etc.). Do not lossy-decode.
        String::from_utf8(decoded).ok()
    }

    fn oversized_payload_result(len: usize) -> IngestionScanResult {
        IngestionScanResult {
            safe: false,
            risk_tier: RiskTier::Critical,
            risk_score: 1.0,
            threats: vec![IngestionThreat::UnknownHighRisk],
            findings: vec![ScanFinding {
                threat: IngestionThreat::UnknownHighRisk,
                signal: "payload_exceeds_max_scan_bytes".into(),
                confidence: 1.0,
                offset: None,
            }],
            details: vec![format!("payload {} > MAX", len)],
            scanned_at: Utc::now(),
            bytes_scanned: len,
        }
    }

    fn finalize_scan(findings: Vec<ScanFinding>, bytes_scanned: usize) -> IngestionScanResult {
        let mut threats: Vec<_> = findings.iter().map(|f| f.threat.clone()).collect();
        threats.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        threats.dedup();
        let max_conf = findings.iter().map(|f| f.confidence).fold(0.0_f32, f32::max);
        let risk_score = if findings.is_empty() { 0.0 } else { max_conf.clamp(0.0, 1.0) };
        let has_hard = findings.iter().any(|f| {
            matches!(
                f.threat,
                IngestionThreat::RemoteCodeLoader
                    | IngestionThreat::SerializationGadget
                    | IngestionThreat::ShellProcessSpawn
                    | IngestionThreat::UnknownHighRisk
            ) && f.confidence >= 0.82
        });
        let risk_tier = if risk_score >= 0.90 || findings.iter().any(|f| f.confidence >= 0.95) {
            RiskTier::Critical
        } else if risk_score >= 0.78 || has_hard {
            RiskTier::High
        } else if risk_score >= 0.40 {
            RiskTier::Medium
        } else if risk_score > 0.0 {
            RiskTier::Low
        } else {
            RiskTier::None
        };
        let safe = matches!(risk_tier, RiskTier::None | RiskTier::Low);
        let details: Vec<_> = findings
            .iter()
            .map(|f| format!("{:?} '{}' {:.2}", f.threat, f.signal, f.confidence))
            .collect();
        IngestionScanResult {
            safe,
            risk_tier,
            risk_score,
            threats,
            findings,
            details,
            scanned_at: Utc::now(),
            bytes_scanned,
        }
    }

    /// Unicode Cf (Format) used as keyword splitters. Does **not** strip Zs spaces.
    /// Stripping spaces would glue "eval (" / innocent words into tripwires.
    fn is_cf_format(c: char) -> bool {
        matches!(
            c,
            '\u{00AD}'
                | '\u{0600}'..='\u{0605}'
                | '\u{061C}'
                | '\u{06DD}'
                | '\u{070F}'
                | '\u{08E2}'
                | '\u{180E}'
                | '\u{200B}'..='\u{200F}'
                | '\u{202A}'..='\u{202E}'
                | '\u{2060}'..='\u{2064}'
                | '\u{2066}'..='\u{206F}'
                | '\u{FEFF}'
                | '\u{FFF9}'..='\u{FFFB}'
                | '\u{110BD}'
                | '\u{110CD}'
                | '\u{1D173}'..='\u{1D17A}'
                | '\u{E0001}'
                | '\u{E0020}'..='\u{E007F}'
        )
    }

    fn strip_cf_format(s: &str) -> String {
        s.chars().filter(|c| !Self::is_cf_format(*c)).collect()
    }

    /// Fullwidth ASCII + common Cyrillic lookalikes used to spoof latin identifiers.
    /// Applied to the keyword haystack only (Base64 tokens are extracted from the original).
    fn fold_homoglyph(c: char) -> char {
        let cp = c as u32;
        if (0xFF01..=0xFF5E).contains(&cp) {
            return char::from_u32(cp - 0xFEE0).unwrap_or(c);
        }
        match c {
            '\u{0430}' | '\u{0410}' => 'a',
            '\u{0435}' | '\u{0415}' => 'e',
            '\u{043E}' | '\u{041E}' => 'o',
            '\u{0440}' | '\u{0420}' => 'p',
            '\u{0441}' | '\u{0421}' => 'c',
            '\u{0443}' | '\u{0423}' => 'y',
            '\u{0445}' | '\u{0425}' => 'x',
            '\u{0456}' | '\u{0406}' => 'i',
            '\u{0455}' | '\u{0405}' => 's',
            _ => c,
        }
    }

    fn fold_keyword_haystack(s: &str) -> String {
        Self::strip_cf_format(s)
            .chars()
            .map(Self::fold_homoglyph)
            .collect::<String>()
            .to_lowercase()
    }

    pub fn scan_text(content: &str) -> IngestionScanResult {
        Self::scan_text_depth(content, 0)
    }

    /// `b64_depth` 0 = may decode obvious standalone Base64 once and re-scan.
    /// Depth 1+ never decodes again (no zip-bomb / nested theater).
    fn scan_text_depth(content: &str, b64_depth: u8) -> IngestionScanResult {
        if content.len() > MAX_SCAN_BYTES {
            return Self::oversized_payload_result(content.len());
        }
        let lower = Self::fold_keyword_haystack(content);
        let mut findings = Self::collect_keyword_findings(&lower);

        if b64_depth == 0 {
            for token in Self::obvious_standalone_base64_tokens(content) {
                if let Some(decoded) = Self::decode_b64_utf8_capped(&token) {
                    if decoded == content {
                        continue;
                    }
                    let inner = Self::scan_text_depth(&decoded, 1);
                    for f in inner.findings {
                        findings.push(ScanFinding {
                            threat: f.threat,
                            signal: format!("b64:{}", f.signal),
                            confidence: f.confidence,
                            offset: f.offset,
                        });
                    }
                }
            }
        }

        Self::finalize_scan(findings, content.len())
    }

    pub fn admit_or_block(content: &str) -> Result<IngestionScanResult, MercySecurityError> {
        if content.len() > MAX_SCAN_BYTES {
            return Err(MercySecurityError::PayloadTooLarge(content.len()));
        }
        let result = Self::scan_text(content);
        if !result.safe {
            return Err(MercySecurityError::IngestionBlocked(format!(
                "tier={} score={:.2}",
                result.risk_tier.as_str(),
                result.risk_score
            )));
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    pub action_id: Uuid,
    pub kind: String,
    pub timestamp: DateTime<Utc>,
    pub sandbox_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionGovernor {
    pub profile: ContainmentProfile,
    pub recent_actions: Vec<ActionRecord>,
    pub total_actions: u64,
    pub trips: u32,
}

impl ActionGovernor {
    pub fn new(profile: ContainmentProfile) -> Self {
        Self {
            profile,
            recent_actions: Vec::new(),
            total_actions: 0,
            trips: 0,
        }
    }

    pub fn record_and_check(
        &mut self,
        kind: &str,
        sandbox_id: Option<&str>,
    ) -> Result<(), MercySecurityError> {
        let now = Utc::now();
        self.recent_actions
            .retain(|a| (now - a.timestamp).num_seconds() < 60);
        if self.recent_actions.len() as u32 >= self.profile.max_actions_per_minute {
            self.trips += 1;
            return Err(MercySecurityError::ActionLimitExceeded(format!(
                ">={} actions/min",
                self.profile.max_actions_per_minute
            )));
        }
        if let Some(sid) = sandbox_id {
            if !self.profile.allow_unbounded_sandbox_spawn {
                let mut unique: std::collections::HashSet<&str> = self
                    .recent_actions
                    .iter()
                    .filter_map(|a| a.sandbox_id.as_deref())
                    .collect();
                unique.insert(sid);
                if unique.len() > self.profile.max_concurrent_sandboxes as usize {
                    self.trips += 1;
                    return Err(MercySecurityError::ActionLimitExceeded(format!(
                        "sandbox churn > {}",
                        self.profile.max_concurrent_sandboxes
                    )));
                }
            }
        }
        self.recent_actions.push(ActionRecord {
            action_id: Uuid::new_v4(),
            kind: kind.into(),
            timestamp: now,
            sandbox_id: sandbox_id.map(|s| s.into()),
        });
        self.total_actions += 1;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopedToken {
    pub token_id: Uuid,
    pub scope: String,
    pub expires_at: DateTime<Utc>,
    pub issued_at: DateTime<Utc>,
}

pub struct SecretVault;
impl SecretVault {
    pub fn issue_scoped_token(scope: &str, ttl_secs: i64) -> Result<ScopedToken, MercySecurityError> {
        if ttl_secs <= 0 || ttl_secs > 3600 {
            return Err(MercySecurityError::Internal("ttl must be 1..=3600".into()));
        }
        Ok(ScopedToken {
            token_id: Uuid::new_v4(),
            scope: scope.into(),
            issued_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::seconds(ttl_secs),
        })
    }
    pub fn refuse_long_lived_credential() -> MercySecurityError {
        MercySecurityError::SecretIsolationViolation
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmRefusalPolicy {
    pub real_world_unauthorized_access: bool,
    pub data_exfiltration: bool,
    pub lateral_movement: bool,
    pub credential_theft: bool,
    pub physical_actuation: bool,
    pub wet_lab_synthesis: bool,
}

impl Default for HarmRefusalPolicy {
    fn default() -> Self {
        Self {
            real_world_unauthorized_access: true,
            data_exfiltration: true,
            lateral_movement: true,
            credential_theft: true,
            physical_actuation: true,
            wet_lab_synthesis: true,
        }
    }
}

impl HarmRefusalPolicy {
    pub fn check_action(&self, action_description: &str) -> Result<(), MercySecurityError> {
        let lower = action_description.to_lowercase();
        if self.real_world_unauthorized_access
            && (lower.contains("unauthorized")
                || lower.contains("break out")
                || lower.contains("escape sandbox")
                || lower.contains("gain internet access"))
        {
            return Err(MercySecurityError::HarmRefusalActive);
        }
        if self.data_exfiltration
            && (lower.contains("exfil")
                || lower.contains("steal data")
                || lower.contains("dump credentials"))
        {
            return Err(MercySecurityError::HarmRefusalActive);
        }
        if self.lateral_movement && lower.contains("lateral") {
            return Err(MercySecurityError::HarmRefusalActive);
        }
        if self.credential_theft
            && (lower.contains("harvest credential")
                || lower.contains("steal key")
                || lower.contains("api key"))
        {
            return Err(MercySecurityError::HarmRefusalActive);
        }
        if self.physical_actuation && Self::is_physical_actuation_signal(&lower) {
            return Err(MercySecurityError::HarmRefusalActive);
        }
        if self.wet_lab_synthesis && Self::is_wet_lab_synthesis_signal(&lower) {
            return Err(MercySecurityError::HarmRefusalActive);
        }
        Ok(())
    }

    pub fn is_physical_actuation_signal(lower: &str) -> bool {
        const SIGNALS: &[&str] = &[
            "actuate motor", "drive motor", "command motor", "open valve", "close valve",
            "fire actuator", "engage actuator", "physical actuation", "real-world actuation",
            "unauthorized actuation", "move robot arm", "command joint torque", "release brake",
            "engage end effector", "energize solenoid", "spin propeller", "fire thruster",
            "physical weapon",
        ];
        SIGNALS.iter().any(|s| lower.contains(s))
    }

    pub fn is_wet_lab_synthesis_signal(lower: &str) -> bool {
        const SIGNALS: &[&str] = &[
            "synthesize pathogen",
            "synthesize toxin",
            "synthesize controlled substance",
            "mix reagent for production",
            "dispense live culture",
            "culture pathogen",
            "amplify select agent",
            "order restricted reagent",
            "ship biological agent",
            "real-world synthesis",
            "unauthorized synthesis",
            "wet-lab production run",
            "incubate select agent",
            "aerosolize culture",
            "release biological agent",
            "weaponize pathogen",
        ];
        SIGNALS.iter().any(|s| lower.contains(s))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationEvent {
    pub id: Uuid,
    pub description: String,
    pub allowed: bool,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteHatEvaluationHarness {
    pub profile: ContainmentProfile,
    pub refusal: HarmRefusalPolicy,
    pub governor: ActionGovernor,
    pub log: Vec<EvaluationEvent>,
    pub unified: UnifiedAgentSurface,
    pub audit_chain: Vec<AuditChainStep>,
}

impl WhiteHatEvaluationHarness {
    pub fn new() -> Self {
        Self::with_profile(ContainmentProfile::evaluation())
    }

    pub fn try_action(
        &mut self,
        description: &str,
        involves_external_network: bool,
        involves_code_exec: bool,
        sandbox_id: Option<&str>,
    ) -> Result<(), MercySecurityError> {
        self.refusal.check_action(description)?;
        self.profile.check_network_allowed(involves_external_network)?;
        if involves_code_exec {
            self.profile.check_code_exec_allowed()?;
        }
        self.governor.record_and_check(description, sandbox_id)?;
        self.log.push(EvaluationEvent {
            id: Uuid::new_v4(),
            description: description.into(),
            allowed: true,
            reason: "passed".into(),
            timestamp: Utc::now(),
        });
        Ok(())
    }

    pub fn audit_log(&self) -> &[EvaluationEvent] {
        &self.log
    }
}

impl Default for WhiteHatEvaluationHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MercySecuritySurface {
    pub default_profile: ContainmentProfile,
    pub refusal: HarmRefusalPolicy,
}
impl MercySecuritySurface {
    pub fn new() -> Self {
        Self {
            default_profile: ContainmentProfile::default(),
            refusal: HarmRefusalPolicy::default(),
        }
    }
    pub fn with_domain_profile(profile: ContainmentProfile) -> Self {
        Self {
            default_profile: profile,
            refusal: HarmRefusalPolicy::default(),
        }
    }
    pub fn scan_ingestion(&self, content: &str) -> Result<IngestionScanResult, MercySecurityError> {
        IngestionScanner::admit_or_block(content)
    }
    pub fn issue_scoped_token(&self, scope: &str, ttl_secs: i64) -> Result<ScopedToken, MercySecurityError> {
        SecretVault::issue_scoped_token(scope, ttl_secs)
    }
    pub fn evaluation_harness(&self) -> WhiteHatEvaluationHarness {
        WhiteHatEvaluationHarness::with_profile(self.default_profile.clone())
    }
    pub fn safe_agent_runtime(&self) -> SafeAgentRuntime {
        SafeAgentRuntime::new(self.default_profile.clone())
    }
    pub fn mercy_council_fleet(&self) -> MercyCouncilFleet {
        MercyCouncilFleet::new(self.default_profile.clone())
    }
    pub fn unified_agent_surface(&self) -> UnifiedAgentSurface {
        UnifiedAgentSurface::new(MercyCouncilFleet::new(self.default_profile.clone()))
    }
}
impl Default for MercySecuritySurface {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_actuation_hard_refuse() {
        let p = HarmRefusalPolicy::default();
        assert!(matches!(
            p.check_action("actuate motor on joint 3"),
            Err(MercySecurityError::HarmRefusalActive)
        ));
        assert!(p.check_action("simulate joint trajectory offline").is_ok());
    }

    #[test]
    fn wet_lab_synthesis_hard_refuse() {
        let p = HarmRefusalPolicy::default();
        assert!(matches!(
            p.check_action("synthesize pathogen in wet lab"),
            Err(MercySecurityError::HarmRefusalActive)
        ));
        assert!(matches!(
            p.check_action("order restricted reagent for production"),
            Err(MercySecurityError::HarmRefusalActive)
        ));
        assert!(p.check_action("simulate reaction kinetics offline").is_ok());
        assert!(p.check_action("plan assay protocol on paper").is_ok());
    }

    #[test]
    fn blocks_remote_code_loader() {
        assert!(IngestionScanner::admit_or_block("trust_remote_code=True").is_err());
    }

    #[test]
    fn fixture_benign_model_card_admits() {
        let content = include_str!("../fixtures/benign/model_card_clean.md");
        let r = IngestionScanner::admit_or_block(content);
        assert!(r.is_ok(), "clean model card must admit: {r:?}");
        let r = r.unwrap();
        assert!(matches!(r.risk_tier, RiskTier::None | RiskTier::Low));
    }

    #[test]
    fn fixture_benign_research_notes_admits() {
        let content = include_str!("../fixtures/benign/research_notes_clean.md");
        assert!(IngestionScanner::admit_or_block(content).is_ok());
    }

    #[test]
    fn fixture_benign_safe_python_snippet_admits() {
        let content = include_str!("../fixtures/benign/safe_python_snippet.md");
        assert!(IngestionScanner::admit_or_block(content).is_ok());
    }

    #[test]
    fn fixture_benign_education_protocol_admits() {
        let content = include_str!("../fixtures/benign/education_protocol.md");
        assert!(IngestionScanner::admit_or_block(content).is_ok());
    }

    #[test]
    fn fixture_benign_base64_tend_the_well_admits() {
        let content = include_str!("../fixtures/benign/base64_tend_the_well.md");
        let r = IngestionScanner::admit_or_block(content);
        assert!(r.is_ok(), "benign Base64 fixture must admit: {r:?}");
    }

    #[test]
    fn fixture_should_block_trust_remote_code() {
        let content = include_str!("../fixtures/should_block/trust_remote_code_loader.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
        let scan = IngestionScanner::scan_text(content);
        assert!(scan.risk_tier >= RiskTier::High);
    }

    #[test]
    fn fixture_should_block_hf_combo() {
        let content = include_str!("../fixtures/should_block/hf_combo_remote_config.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
        let scan = IngestionScanner::scan_text(content);
        assert!(scan.risk_tier >= RiskTier::High);
        assert!(scan.threats.contains(&IngestionThreat::UnknownHighRisk)
            || scan.threats.contains(&IngestionThreat::RemoteCodeLoader));
    }

    #[test]
    fn fixture_should_block_pickle_gadget() {
        let content = include_str!("../fixtures/should_block/pickle_gadget.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
    }

    #[test]
    fn fixture_should_block_obfuscated_exec() {
        let content = include_str!("../fixtures/should_block/obfuscated_exec_pattern.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
        let scan = IngestionScanner::scan_text(content);
        assert!(scan.risk_tier >= RiskTier::High);
    }

    #[test]
    fn fixture_should_block_shell_network_combo() {
        let content = include_str!("../fixtures/should_block/shell_network_combo.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
    }

    #[test]
    fn fixture_should_block_pem_private_key() {
        let content = include_str!("../fixtures/should_block/pem_private_key_marker.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
        let scan = IngestionScanner::scan_text(content);
        assert!(scan.threats.contains(&IngestionThreat::CredentialHarvestPattern));
    }

    #[test]
    fn fixture_should_block_template_jinja() {
        let content = include_str!("../fixtures/should_block/template_jinja_injection.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
        let scan = IngestionScanner::scan_text(content);
        assert!(scan.threats.contains(&IngestionThreat::TemplateInjection));
    }

    #[test]
    fn fixture_should_block_network_callback() {
        let content = include_str!("../fixtures/should_block/network_callback_marker.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
        let scan = IngestionScanner::scan_text(content);
        assert!(scan.threats.contains(&IngestionThreat::NetworkCallback));
    }

    #[test]
    fn fixture_should_block_dataset_loading_script() {
        let content = include_str!("../fixtures/should_block/dataset_loading_script.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
    }

    #[test]
    fn fixture_should_block_credential_hf_token() {
        let content = include_str!("../fixtures/should_block/credential_hf_token.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
        let scan = IngestionScanner::scan_text(content);
        assert!(scan.threats.contains(&IngestionThreat::CredentialHarvestPattern));
    }

    #[test]
    fn fixture_should_block_yaml_unsafe_load() {
        let content = include_str!("../fixtures/should_block/yaml_unsafe_load.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
        let scan = IngestionScanner::scan_text(content);
        assert!(scan.threats.contains(&IngestionThreat::SerializationGadget));
    }

    #[test]
    fn fixture_docs_api_key_fp_probe() {
        let content = include_str!("../fixtures/benign/docs_mention_api_key.md");
        let scan = IngestionScanner::scan_text(content);
        assert!(scan.risk_tier <= RiskTier::Medium, "must not escalate to High/Critical");
    }

    #[test]
    fn base64_trust_remote_code_without_decoder_token_blocks() {
        // Standard Base64 of `trust_remote_code` — no decoder token in the paste.
        let token = "dHJ1c3RfcmVtb3RlX2NvZGU=";
        assert!(IngestionScanner::admit_or_block(token).is_err());
        let scan = IngestionScanner::scan_text(token);
        assert!(scan
            .findings
            .iter()
            .any(|f| f.signal.contains("trust_remote_code")));
    }

    #[test]
    fn base64_benign_prose_admits() {
        // Standard Base64 of "tend the well" — obvious token, no tripwire after decode.
        let token = "dGVuZCB0aGUgd2VsbA==";
        let r = IngestionScanner::admit_or_block(token);
        assert!(r.is_ok(), "benign Base64 prose must admit: {r:?}");
    }

    #[test]
    fn nested_base64_is_not_decoded_twice() {
        // One-level rescan only. Outer is Base64 of the Base64 of trust_remote_code.
        let double = "dEhKMWMzUmZjbVZibTNSbFgyTnZaR1U9";
        assert!(
            IngestionScanner::admit_or_block(double).is_ok(),
            "nested Base64 is not theater-decoded"
        );
    }

    #[test]
    fn cf_format_split_trust_remote_code_blocks() {
        assert!(IngestionScanner::admit_or_block("trust_\u{200b}remote_code").is_err());
        assert!(IngestionScanner::admit_or_block("trust_\u{200c}remote_code").is_err());
        assert!(IngestionScanner::admit_or_block("trust_\u{200d}remote_code").is_err());
        assert!(IngestionScanner::admit_or_block("trust_\u{feff}remote_code").is_err());
    }

    #[test]
    fn zwsp_inside_flow_state_prose_admits() {
        let prose = "Classroom notes on flo\u{200b}w state during the mercy tick.";
        let r = IngestionScanner::admit_or_block(prose);
        assert!(r.is_ok(), "ZWSP inside benign 'flow state' prose must admit: {r:?}");
    }

    #[test]
    fn whitespace_split_trust_remote_code_blocks() {
        assert!(IngestionScanner::admit_or_block("trust_ remote_code").is_err());
        assert!(IngestionScanner::admit_or_block("trust_\tremote_code").is_err());
        assert!(IngestionScanner::admit_or_block("trust_\nremote_code").is_err());
        assert!(IngestionScanner::admit_or_block("trust_\u{00a0}remote_code").is_err());
        assert!(IngestionScanner::admit_or_block("pickle. loads").is_err());
    }

    #[test]
    fn subprocess_space_split_still_admits() {
        // Single-token signals do not skip spaces: "sub process" ≠ subprocess.
        let r = IngestionScanner::admit_or_block("Classroom notes on a sub process in the lab.");
        assert!(r.is_ok(), "must not glue subprocess: {r:?}");
    }

    #[test]
    fn academic_eval_metrics_paren_still_admits() {
        let r = IngestionScanner::admit_or_block(
            "The academic eval (metrics) discussion stays offline. No remote loaders.",
        );
        assert!(r.is_ok(), "must not glue eval(: {r:?}");
    }

    #[test]
    fn fullwidth_trust_remote_code_blocks() {
        assert!(IngestionScanner::admit_or_block("ｔｒｕｓｔ＿ｒｅｍｏｔｅ＿ｃｏｄｅ").is_err());
    }

    #[test]
    fn cyrillic_homoglyph_trust_remote_code_blocks() {
        assert!(IngestionScanner::admit_or_block("tru\u{0455}t_remote_code").is_err());
        assert!(IngestionScanner::admit_or_block("trust_remote_\u{0441}ode").is_err());
        assert!(IngestionScanner::admit_or_block("pickle.l\u{043E}ads").is_err());
    }

    #[test]
    fn benign_cyrillic_prose_admits() {
        let r = IngestionScanner::admit_or_block(
            "Заметки о милосердии и classroom flow state. No remote loaders.",
        );
        assert!(r.is_ok(), "benign Cyrillic prose must admit: {r:?}");
    }
}
