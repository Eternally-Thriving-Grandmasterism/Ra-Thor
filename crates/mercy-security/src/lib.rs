//! # Mercy-Security — White-Hat AGSi Defense (v14.15.4)
//!
//! Defensive surface designed against the July 2026 OpenAI → Hugging Face
//! autonomous agent breach class and related AI supply-chain / containment failures.
//!
//! Core capabilities:
//! - Hard containment profiles (network, code execution, credential visibility)
//! - Ingestion scanner for datasets / models / configs (remote-code + template injection)
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
            allow_unrestricted_network: false, // still blocked
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
// 2. Ingestion Scanner (dataset / model / config)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IngestionThreat {
    RemoteCodeLoader,
    TemplateInjection,
    SuspiciousExecutablePayload,
    CredentialHarvestPattern,
    UnknownHighRisk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionScanResult {
    pub safe: bool,
    pub threats: Vec<IngestionThreat>,
    pub details: Vec<String>,
    pub scanned_at: DateTime<Utc>,
}

/// Lightweight pattern scanner for the two primary HF July 2026 vectors
/// (remote-code dataset loader + template injection) plus common follow-ons.
pub struct IngestionScanner;

impl IngestionScanner {
    pub fn scan_text(content: &str) -> IngestionScanResult {
        let lower = content.to_lowercase();
        let mut threats = Vec::new();
        let mut details = Vec::new();

        // Remote-code dataset loader patterns
        let remote_code_signals = [
            "trust_remote_code",
            "remote_code=true",
            "__import__",
            "exec(",
            "eval(",
            "subprocess",
            "os.system",
            "pickle.loads",
            "torch.load",
            "joblib.load",
        ];
        for sig in remote_code_signals {
            if lower.contains(sig) {
                threats.push(IngestionThreat::RemoteCodeLoader);
                details.push(format!("remote-code signal: {}", sig));
                break;
            }
        }

        // Template injection / config injection
        let template_signals = [
            "{{",
            "{%",
            "${",
            "jinja",
            "template.render",
            "format(",
            "%s" ,
        ];
        // Only flag template injection when combined with code-like context
        let has_template = template_signals.iter().any(|s| lower.contains(s));
        let has_code_context = lower.contains("exec")
            || lower.contains("eval")
            || lower.contains("__class__")
            || lower.contains("__builtins__");
        if has_template && has_code_context {
            threats.push(IngestionThreat::TemplateInjection);
            details.push("template injection + code context detected".into());
        }

        // Credential harvest patterns
        if lower.contains("aws_secret")
            || lower.contains("api_key")
            || lower.contains("private_key")
            || lower.contains("bearer ")
        {
            threats.push(IngestionThreat::CredentialHarvestPattern);
            details.push("credential-like pattern present".into());
        }

        let safe = threats.is_empty();
        IngestionScanResult {
            safe,
            threats,
            details,
            scanned_at: Utc::now(),
        }
    }

    /// Hard gate: returns error if any threat is present.
    pub fn admit_or_block(content: &str) -> Result<IngestionScanResult, MercySecurityError> {
        let result = Self::scan_text(content);
        if !result.safe {
            return Err(MercySecurityError::IngestionBlocked(format!(
                "threats={:?} details={:?}",
                result.threats, result.details
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

    /// Record an action and enforce rate / volume limits.
    pub fn record_and_check(&mut self, kind: &str, sandbox_id: Option<&str>) -> Result<(), MercySecurityError> {
        let now = Utc::now();
        // Prune older than 60s
        self.recent_actions
            .retain(|a| (now - a.timestamp).num_seconds() < 60);

        if self.recent_actions.len() as u32 >= self.profile.max_actions_per_minute {
            self.trips += 1;
            return Err(MercySecurityError::ActionLimitExceeded(format!(
                ">{} actions/min (trip #{})",
                self.profile.max_actions_per_minute, self.trips
            )));
        }

        // Detect self-migrating / high-churn sandbox pattern (HF incident style)
        if let Some(sid) = sandbox_id {
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
            let _ = sid;
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

/// Agents receive only short-lived, scoped tokens. Long-lived secrets never leave the vault.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopedToken {
    pub token_id: Uuid,
    pub scope: String,
    pub expires_at: DateTime<Utc>,
    pub issued_at: DateTime<Utc>,
}

pub struct SecretVault;

impl SecretVault {
    /// Issue a short-lived token. Never returns the underlying long-lived secret.
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

    /// Explicitly refuse any request that would hand a long-lived credential to an agent.
    pub fn refuse_long_lived_credential() -> MercySecurityError {
        MercySecurityError::SecretIsolationViolation
    }
}

// =============================================================================
// 5. Never-Disable Harm Refusals
// =============================================================================

/// Even when running cyber-capability evaluations, real-world unauthorized access,
/// data exfiltration, and lateral movement remain refused.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmRefusalPolicy {
    pub real_world_unauthorized_access: bool, // always true
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

    /// Attempt an evaluation action under full white-hat constraints.
    pub fn try_action(
        &mut self,
        description: &str,
        involves_external_network: bool,
        involves_code_exec: bool,
        sandbox_id: Option<&str>,
    ) -> Result<(), MercySecurityError> {
        // 1. Harm refusals never disabled
        self.refusal.check_action(description)?;

        // 2. Containment
        self.profile.check_network_allowed(involves_external_network)?;
        if involves_code_exec {
            self.profile.check_code_exec_allowed()?;
        }

        // 3. Action governor
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

/// Single entry point for Ra-Thor / ONE Organism / PATSAGi to consume.
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
    fn allows_clean_content() {
        let content = "This is a normal markdown description of a model.";
        assert!(IngestionScanner::admit_or_block(content).is_ok());
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
