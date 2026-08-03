//! # Mercy-Security — White-Hat AGSi Defense (v14.15.5)
//!
//! Domain profiles: education · research · enterprise · creative · robotics · biomedical
//! Physical actuation + wet-lab synthesis hard-refuse under HarmRefusalPolicy.
//! TOLC 8 + PATSAGi | AG-SML v1.0 | Contact: info@Rathor.ai

mod domain_profiles;
mod safe_agent_runtime;
mod mercy_council_fleet;
mod unified_agent_surface;

pub use safe_agent_runtime::{
    AgentActionReceipt, AgentActionRequest, SafeAgentRuntime, AGENT_TOKEN_MAX_TTL_SECS,
};
pub use mercy_council_fleet::{
    AgentIsolationLevel, FleetAgentSlot, FleetRiskTier, FleetSecuritySignal, MercyCouncilFleet,
    DEFAULT_PER_AGENT_BUDGET_SHARE, FLEET_PROGRESSIVE_VALENCE_FLOOR,
};
pub use unified_agent_surface::{UnifiedAgentSurface, GOVERNOR_TRIPS_PER_ISOLATION_STEP};
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
            if lower.contains(sig) {
                findings.push(ScanFinding {
                    threat: threat.clone(),
                    signal: (*sig).into(),
                    confidence: *conf,
                    offset: lower.find(sig),
                });
            }
        }
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
                details: vec![format!("payload {} > MAX", content.len())],
                scanned_at: Utc::now(),
                bytes_scanned: content.len(),
            };
        }
        let lower = content.to_lowercase();
        let mut findings = Vec::new();
        Self::match_signals(&lower, Self::remote_code_signals(), IngestionThreat::RemoteCodeLoader, &mut findings);
        Self::match_signals(&lower, Self::serialization_gadget_signals(), IngestionThreat::SerializationGadget, &mut findings);
        Self::match_signals(&lower, Self::shell_spawn_signals(), IngestionThreat::ShellProcessSpawn, &mut findings);
        Self::match_signals(&lower, Self::network_callback_signals(), IngestionThreat::NetworkCallback, &mut findings);
        Self::match_signals(&lower, Self::obfuscation_signals(), IngestionThreat::ObfuscatedPayload, &mut findings);
        Self::match_signals(&lower, Self::dataset_config_signals(), IngestionThreat::DatasetConfigInjection, &mut findings);
        Self::match_signals(&lower, Self::credential_signals(), IngestionThreat::CredentialHarvestPattern, &mut findings);
        Self::match_signals(&lower, Self::template_signals(), IngestionThreat::TemplateInjection, &mut findings);

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
    /// Hard refuse unauthorized real-world wet-lab synthesis / reagent handling.
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

    /// Unauthorized real-world wet-lab / synthesis / reagent patterns.
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

    // ── Public fixture corpus (Tier A admit gate) ──────────────────────────

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
        // Documented as Low-or-Medium FP probe; policy blocks Medium+.
        let content = include_str!("../fixtures/benign/docs_mention_api_key.md");
        let scan = IngestionScanner::scan_text(content);
        assert!(scan.risk_tier <= RiskTier::Medium, "must not escalate to High/Critical");
        // Whether admit or block depends on exact threshold; both are acceptable for this probe.
    }
}
