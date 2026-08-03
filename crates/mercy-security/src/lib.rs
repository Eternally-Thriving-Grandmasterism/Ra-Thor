//! # Mercy-Security — White-Hat AGSi Defense (v14.15.5)
//!
//! Defensive surface designed against the July 2026 OpenAI → Hugging Face
//! autonomous agent breach class and related AI supply-chain / containment failures.
//!
//! Core capabilities:
//! - Hard containment profiles (network, code execution, credential visibility)
//! - Named domain profiles: research / enterprise / education / creative
//! - Deep multi-layer IngestionScanner (remote-code, template injection, gadgets, C2, obfuscation)
//! - Autonomous action governor (rate limits, sandbox churn) — trips first on volume
//! - SafeAgentRuntime facade (ordered gates + short-lived scoped tokens)
//! - Secret isolation (agents never receive long-lived credentials)
//! - Never-disable real-world harm refusals (even in evaluation mode)
//! - White-hat evaluation harness under strict sandbox + full audit log
//!
//! Policy (unattended ingest): block Medium + High + Critical (only None/Low admitted).
//! TOLC 8 + PATSAGi aligned | AG-SML v1.0 | Contact: info@Rathor.ai

mod domain_profiles;
mod safe_agent_runtime;

pub use safe_agent_runtime::{
    AgentActionReceipt, AgentActionRequest, SafeAgentRuntime, AGENT_TOKEN_MAX_TTL_SECS,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Canonical mercy valence floor for security-adjacent decisions.
pub const MERCY_VALENCE_FLOOR: f64 = 0.999;

/// Maximum payload size accepted by the ingestion scanner (4 MiB).
pub const MAX_SCAN_BYTES: usize = 4 * 1024 * 1024;

// =============================================================================
// Error surface
// =============================================================================

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

// =============================================================================
// 1. Containment Profiles
// =============================================================================

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
            allow_unrestricted_network: false,
            allow_remote_code_execution: false,
            allow_long_lived_credentials: false,
            allow_unbounded_sandbox_spawn: false,
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

// Domain presets live in domain_profiles.rs
// Safe agent runtime lives in safe_agent_runtime.rs

// =============================================================================
// 2. Deep Ingestion Scanner
// =============================================================================

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
        &[
            ("trust_remote_code", 0.98),
            ("remote_code=true", 0.97),
            ("remote_code = true", 0.97),
            ("trust_remote_code=true", 0.99),
            ("__import__", 0.85),
            ("importlib.import_module", 0.88),
            ("importlib.util", 0.80),
            ("exec(", 0.92),
            ("eval(", 0.92),
            ("compile(", 0.78),
            ("__builtins__", 0.90),
            ("__class__", 0.72),
            ("globals()", 0.70),
            ("locals()", 0.65),
            ("getattr(", 0.55),
            ("setattr(", 0.55),
            ("ctypes", 0.75),
            ("cffi", 0.70),
            ("loading_script", 0.88),
            ("dataset_script", 0.85),
            ("_generate_examples", 0.70),
            ("huggingface_hub.hf_api", 0.60),
        ]
    }

    fn serialization_gadget_signals() -> &'static [(&'static str, f32)] {
        &[
            ("pickle.loads", 0.95),
            ("pickle.load", 0.93),
            ("_pickle", 0.80),
            ("joblib.load", 0.90),
            ("torch.load", 0.88),
            ("torch.jit.load", 0.82),
            ("onnxruntime", 0.50),
            ("numpy.load", 0.55),
            ("yaml.load(", 0.85),
            ("yaml.unsafe_load", 0.97),
            ("marshal.loads", 0.90),
            ("shelve.open", 0.70),
            ("dill.loads", 0.92),
            ("cloudpickle", 0.90),
        ]
    }

    fn shell_spawn_signals() -> &'static [(&'static str, f32)] {
        &[
            ("subprocess", 0.88),
            ("os.system", 0.95),
            ("os.popen", 0.92),
            ("shell=true", 0.93),
            ("shell = true", 0.93),
            ("popen(", 0.80),
            ("commands.getoutput", 0.85),
            ("pty.spawn", 0.90),
            ("/bin/sh", 0.85),
            ("/bin/bash", 0.85),
            ("cmd.exe", 0.85),
            ("powershell", 0.80),
        ]
    }

    fn network_callback_signals() -> &'static [(&'static str, f32)] {
        &[
            ("reverse shell", 0.95),
            ("/dev/tcp/", 0.92),
            ("socket.connect", 0.80),
            ("requests.post", 0.55),
            ("urllib.request", 0.55),
            ("http.client", 0.55),
            ("webhook", 0.50),
            ("ngrok", 0.70),
            ("discord.com/api/webhooks", 0.85),
            ("pastebin.com", 0.70),
            ("transfer.sh", 0.75),
            ("curl ", 0.45),
            ("wget ", 0.45),
            ("nc -e", 0.95),
            ("ncat ", 0.90),
            ("socat ", 0.88),
        ]
    }

    fn obfuscation_signals() -> &'static [(&'static str, f32)] {
        &[
            ("base64.b64decode", 0.80),
            ("base64.b64decode(", 0.85),
            ("codecs.decode", 0.70),
            ("bytes.fromhex", 0.75),
            ("chr(", 0.40),
            ("ord(", 0.35),
            ("eval(compile(", 0.97),
            ("exec(compile(", 0.97),
            ("__reduce__", 0.90),
            ("__setstate__", 0.85),
            ("marshal.loads(base64", 0.95),
        ]
    }

    fn template_signals() -> &'static [(&'static str, f32)] {
        &[
            ("{{", 0.40),
            ("{%", 0.50),
            ("${", 0.35),
            ("jinja", 0.70),
            ("jinja2", 0.75),
            ("template.render", 0.85),
            ("environment.from_string", 0.88),
            ("mako.template", 0.85),
            ("string.template", 0.60),
        ]
    }

    fn dataset_config_signals() -> &'static [(&'static str, f32)] {
        &[
            ("dataset_infos.json", 0.60),
            ("loading_script", 0.85),
            ("_urls", 0.40),
            ("download_and_extract", 0.55),
            ("dl_manager", 0.65),
            ("builder_config", 0.55),
            ("gen_kwargs", 0.50),
            ("script_version", 0.45),
            ("features=datasets", 0.35),
        ]
    }

    fn credential_signals() -> &'static [(&'static str, f32)] {
        &[
            ("aws_secret_access_key", 0.95),
            ("aws_access_key_id", 0.90),
            ("api_key", 0.52),
            ("apikey", 0.52),
            ("private_key", 0.85),
            ("-----begin private key-----", 0.98),
            ("-----begin rsa private key-----", 0.98),
            ("bearer ", 0.75),
            ("authorization: bearer", 0.85),
            ("x-api-key", 0.80),
            ("secret_key", 0.70),
            ("client_secret", 0.85),
            ("hf_token", 0.90),
            ("huggingface_token", 0.90),
            ("openai_api_key", 0.92),
            ("anthropic_api_key", 0.92),
        ]
    }

    fn match_signals(
        lower: &str,
        original: &str,
        table: &[(&str, f32)],
        threat: IngestionThreat,
        findings: &mut Vec<ScanFinding>,
    ) {
        for (sig, conf) in table {
            if lower.contains(sig) {
                let offset = lower.find(sig);
                findings.push(ScanFinding {
                    threat: threat.clone(),
                    signal: (*sig).into(),
                    confidence: *conf,
                    offset,
                });
            }
        }
        let _ = original;
    }

    pub fn scan_text(content: &str) -> IngestionScanResult {
        if content.len() > MAX_SCAN_BYTES {
            return IngestionScanResult {
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
                details: vec![format!(
                    "payload {} bytes > MAX_SCAN_BYTES={}",
                    content.len(),
                    MAX_SCAN_BYTES
                )],
                scanned_at: Utc::now(),
                bytes_scanned: content.len(),
            };
        }

        let lower = content.to_lowercase();
        let mut findings: Vec<ScanFinding> = Vec::new();

        Self::match_signals(&lower, content, Self::remote_code_signals(), IngestionThreat::RemoteCodeLoader, &mut findings);
        Self::match_signals(&lower, content, Self::serialization_gadget_signals(), IngestionThreat::SerializationGadget, &mut findings);
        Self::match_signals(&lower, content, Self::shell_spawn_signals(), IngestionThreat::ShellProcessSpawn, &mut findings);
        Self::match_signals(&lower, content, Self::network_callback_signals(), IngestionThreat::NetworkCallback, &mut findings);
        Self::match_signals(&lower, content, Self::obfuscation_signals(), IngestionThreat::ObfuscatedPayload, &mut findings);
        Self::match_signals(&lower, content, Self::dataset_config_signals(), IngestionThreat::DatasetConfigInjection, &mut findings);
        Self::match_signals(&lower, content, Self::credential_signals(), IngestionThreat::CredentialHarvestPattern, &mut findings);

        let mut template_hits = Vec::new();
        Self::match_signals(&lower, content, Self::template_signals(), IngestionThreat::TemplateInjection, &mut template_hits);

        let has_code_context = findings.iter().any(|f| {
            matches!(
                f.threat,
                IngestionThreat::RemoteCodeLoader
                    | IngestionThreat::ShellProcessSpawn
                    | IngestionThreat::SerializationGadget
                    | IngestionThreat::ObfuscatedPayload
            ) && f.confidence >= 0.70
        });

        if has_code_context {
            for mut t in template_hits {
                t.confidence = (t.confidence + 0.35).min(0.99);
                findings.push(t);
            }
        } else {
            for t in template_hits {
                if t.confidence >= 0.70 {
                    findings.push(t);
                }
            }
        }

        let has_remote = findings.iter().any(|f| f.threat == IngestionThreat::RemoteCodeLoader && f.confidence >= 0.70);
        let has_dataset_cfg = findings.iter().any(|f| f.threat == IngestionThreat::DatasetConfigInjection && f.confidence >= 0.55);
        let has_network = findings.iter().any(|f| f.threat == IngestionThreat::NetworkCallback && f.confidence >= 0.70);
        let has_shell = findings.iter().any(|f| f.threat == IngestionThreat::ShellProcessSpawn && f.confidence >= 0.80);
        let has_obfuscation = findings.iter().any(|f| f.threat == IngestionThreat::ObfuscatedPayload && f.confidence >= 0.75);

        if has_remote && has_dataset_cfg {
            findings.push(ScanFinding {
                threat: IngestionThreat::UnknownHighRisk,
                signal: "combo:remote_code+dataset_config".into(),
                confidence: 0.96,
                offset: None,
            });
        }
        if has_remote && has_network {
            findings.push(ScanFinding {
                threat: IngestionThreat::UnknownHighRisk,
                signal: "combo:remote_code+network_callback".into(),
                confidence: 0.97,
                offset: None,
            });
        }
        if has_shell && has_network {
            findings.push(ScanFinding {
                threat: IngestionThreat::UnknownHighRisk,
                signal: "combo:shell+network_callback".into(),
                confidence: 0.98,
                offset: None,
            });
        }
        if has_obfuscation && (has_remote || has_shell) {
            findings.push(ScanFinding {
                threat: IngestionThreat::UnknownHighRisk,
                signal: "combo:obfuscation+code_exec".into(),
                confidence: 0.97,
                offset: None,
            });
        }

        let mut threats: Vec<IngestionThreat> = findings.iter().map(|f| f.threat.clone()).collect();
        threats.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        threats.dedup();

        let max_conf = findings.iter().map(|f| f.confidence).fold(0.0_f32, f32::max);
        let sum_conf: f32 = findings.iter().map(|f| f.confidence).sum();
        let risk_score = if findings.is_empty() {
            0.0
        } else {
            (0.65 * max_conf + 0.35 * (sum_conf / findings.len() as f32)).clamp(0.0, 1.0)
        };

        let has_hard_exec = findings.iter().any(|f| {
            matches!(
                f.threat,
                IngestionThreat::RemoteCodeLoader
                    | IngestionThreat::SerializationGadget
                    | IngestionThreat::ShellProcessSpawn
                    | IngestionThreat::UnknownHighRisk
            ) && f.confidence >= 0.82
        });
        let has_combo = findings
            .iter()
            .any(|f| f.threat == IngestionThreat::UnknownHighRisk && f.confidence >= 0.95);

        let risk_tier = if risk_score >= 0.90 || findings.iter().any(|f| f.confidence >= 0.95) || has_combo {
            RiskTier::Critical
        } else if risk_score >= 0.78 || has_hard_exec {
            RiskTier::High
        } else if risk_score >= 0.40 {
            RiskTier::Medium
        } else if risk_score > 0.0 {
            RiskTier::Low
        } else {
            RiskTier::None
        };

        let safe = matches!(risk_tier, RiskTier::None | RiskTier::Low);

        let details: Vec<String> = findings
            .iter()
            .map(|f| {
                format!(
                    "{:?} signal='{}' conf={:.2} offset={:?}",
                    f.threat, f.signal, f.confidence, f.offset
                )
            })
            .collect();

        IngestionScanResult {
            safe,
            risk_tier,
            risk_score,
            threats,
            findings,
            details,
            scanned_at: Utc::now(),
            bytes_scanned: content.len(),
        }
    }

    pub fn admit_or_block(content: &str) -> Result<IngestionScanResult, MercySecurityError> {
        if content.len() > MAX_SCAN_BYTES {
            return Err(MercySecurityError::PayloadTooLarge(content.len()));
        }
        let result = Self::scan_text(content);
        if !result.safe {
            return Err(MercySecurityError::IngestionBlocked(format!(
                "tier={} score={:.2} threats={:?} findings={}",
                result.risk_tier.as_str(),
                result.risk_score,
                result.threats,
                result.findings.len()
            )));
        }
        Ok(result)
    }

    pub fn admit_or_block_critical_only(content: &str) -> Result<IngestionScanResult, MercySecurityError> {
        if content.len() > MAX_SCAN_BYTES {
            return Err(MercySecurityError::PayloadTooLarge(content.len()));
        }
        let result = Self::scan_text(content);
        if result.risk_tier == RiskTier::Critical {
            return Err(MercySecurityError::IngestionBlocked(format!(
                "critical tier score={:.2} threats={:?}",
                result.risk_score, result.threats
            )));
        }
        Ok(result)
    }
}

// =============================================================================
// 3. Autonomous Action Governor
// =============================================================================

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

    pub fn from_domain_research() -> Self {
        Self::new(ContainmentProfile::research())
    }

    pub fn from_domain_enterprise() -> Self {
        Self::new(ContainmentProfile::enterprise())
    }

    pub fn from_domain_education() -> Self {
        Self::new(ContainmentProfile::education())
    }

    pub fn record_and_check(&mut self, kind: &str, sandbox_id: Option<&str>) -> Result<(), MercySecurityError> {
        let now = Utc::now();
        self.recent_actions
            .retain(|a| (now - a.timestamp).num_seconds() < 60);

        if self.recent_actions.len() as u32 >= self.profile.max_actions_per_minute {
            self.trips += 1;
            return Err(MercySecurityError::ActionLimitExceeded(format!(
                ">={} actions/min (trip #{})",
                self.profile.max_actions_per_minute, self.trips
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
                        "sandbox churn exceeded max_concurrent_sandboxes={}",
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

// =============================================================================
// 4. Secret Isolation
// =============================================================================

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
            return Err(MercySecurityError::Internal(
                "ttl must be 1..=3600 seconds".into(),
            ));
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

// =============================================================================
// 5. Never-Disable Harm Refusals
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmRefusalPolicy {
    pub real_world_unauthorized_access: bool,
    pub data_exfiltration: bool,
    pub lateral_movement: bool,
    pub credential_theft: bool,
}

impl Default for HarmRefusalPolicy {
    fn default() -> Self {
        Self {
            real_world_unauthorized_access: true,
            data_exfiltration: true,
            lateral_movement: true,
            credential_theft: true,
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
        Ok(())
    }
}

// =============================================================================
// 6. White-Hat Evaluation Harness
// =============================================================================

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
}

impl WhiteHatEvaluationHarness {
    pub fn new() -> Self {
        let profile = ContainmentProfile::evaluation();
        let governor = ActionGovernor::new(profile.clone());
        Self {
            profile,
            refusal: HarmRefusalPolicy::default(),
            governor,
            log: Vec::new(),
        }
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
            reason: "passed white-hat gates".into(),
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

// =============================================================================
// Top-level facade
// =============================================================================

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

    pub fn scan_ingestion_report(&self, content: &str) -> IngestionScanResult {
        IngestionScanner::scan_text(content)
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
    fn blocks_remote_code_loader() {
        let content = "dataset = load_dataset('x', trust_remote_code=True)";
        let err = IngestionScanner::admit_or_block(content);
        assert!(matches!(err, Err(MercySecurityError::IngestionBlocked(_))));
    }

    #[test]
    fn allows_clean_content() {
        let content = "This is a normal markdown description of a model for image classification.";
        let r = IngestionScanner::admit_or_block(content);
        assert!(r.is_ok());
    }

    #[test]
    fn action_governor_trips_on_rate() {
        let mut gov = ActionGovernor::new(ContainmentProfile {
            max_actions_per_minute: 3,
            ..ContainmentProfile::default()
        });
        assert!(gov.record_and_check("a", None).is_ok());
        assert!(gov.record_and_check("b", None).is_ok());
        assert!(gov.record_and_check("c", None).is_ok());
        assert!(matches!(
            gov.record_and_check("d", None),
            Err(MercySecurityError::ActionLimitExceeded(_))
        ));
    }

    #[test]
    fn sandbox_churn_trips_at_max_plus_one() {
        let mut gov = ActionGovernor::new(ContainmentProfile {
            max_concurrent_sandboxes: 2,
            max_actions_per_minute: 100,
            ..ContainmentProfile::default()
        });
        assert!(gov.record_and_check("a", Some("s1")).is_ok());
        assert!(gov.record_and_check("b", Some("s2")).is_ok());
        assert!(matches!(
            gov.record_and_check("c", Some("s3")),
            Err(MercySecurityError::ActionLimitExceeded(_))
        ));
    }

    #[test]
    fn domain_governor_constructors() {
        let e = ActionGovernor::from_domain_education();
        assert_eq!(e.profile.max_actions_per_minute, 30);
        assert_eq!(e.profile.max_concurrent_sandboxes, 2);
    }

    #[test]
    fn harm_refusal_blocks_sandbox_escape() {
        let policy = HarmRefusalPolicy::default();
        let err = policy.check_action("attempt to escape sandbox and gain internet access");
        assert!(matches!(err, Err(MercySecurityError::HarmRefusalActive)));
    }
}
