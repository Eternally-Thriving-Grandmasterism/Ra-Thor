//! Safe agent runtime — ActionGovernor + SecretVault under ContainmentProfiles.
//!
//! Gate order for a proposed tool action:
//! 1. HarmRefusalPolicy (never disabled)
//! 2. ContainmentProfile (network / code-exec)
//! 3. ActionGovernor (rate limit + sandbox churn) — **trips first** on volume abuse
//! 4. Optional short-lived scoped token issue (never long-lived)
//!
//! Contact: info@Rathor.ai

use super::{
    ActionGovernor, ContainmentProfile, HarmRefusalPolicy, MercySecurityError, ScopedToken,
    SecretVault,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Maximum TTL for agent-facing scoped tokens (15 minutes).
pub const AGENT_TOKEN_MAX_TTL_SECS: i64 = 900;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentActionRequest {
    pub description: String,
    pub involves_external_network: bool,
    pub involves_code_exec: bool,
    pub sandbox_id: Option<String>,
    /// If true, issue a short-lived scoped token after gates pass.
    pub request_scoped_token: bool,
    pub token_scope: Option<String>,
    pub token_ttl_secs: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentActionReceipt {
    pub action_id: Uuid,
    pub allowed: bool,
    pub description: String,
    pub profile_name: String,
    pub governor_trips: u32,
    pub scoped_token: Option<ScopedToken>,
    pub timestamp: DateTime<Utc>,
}

/// Production-facing safe agent envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeAgentRuntime {
    pub profile: ContainmentProfile,
    pub refusal: HarmRefusalPolicy,
    pub governor: ActionGovernor,
    pub actions_allowed: u64,
    pub actions_denied: u64,
}

impl SafeAgentRuntime {
    pub fn new(profile: ContainmentProfile) -> Self {
        let governor = ActionGovernor::new(profile.clone());
        Self {
            profile,
            refusal: HarmRefusalPolicy::default(),
            governor,
            actions_allowed: 0,
            actions_denied: 0,
        }
    }

    pub fn research() -> Self {
        Self::new(ContainmentProfile::research())
    }

    pub fn enterprise() -> Self {
        Self::new(ContainmentProfile::enterprise())
    }

    pub fn education() -> Self {
        Self::new(ContainmentProfile::education())
    }

    pub fn creative() -> Self {
        Self::new(ContainmentProfile::creative_content_only())
    }

    /// Primary entry: run ordered gates; governor enforces rate/churn after policy checks.
    pub fn try_agent_action(
        &mut self,
        req: &AgentActionRequest,
    ) -> Result<AgentActionReceipt, MercySecurityError> {
        // 1. Never-disable harm refusals
        if let Err(e) = self.refusal.check_action(&req.description) {
            self.actions_denied = self.actions_denied.saturating_add(1);
            return Err(e);
        }
        // 2. Containment
        if let Err(e) = self.profile.check_network_allowed(req.involves_external_network) {
            self.actions_denied = self.actions_denied.saturating_add(1);
            return Err(e);
        }
        if req.involves_code_exec {
            if let Err(e) = self.profile.check_code_exec_allowed() {
                self.actions_denied = self.actions_denied.saturating_add(1);
                return Err(e);
            }
        }
        // 3. Governor — rate + sandbox churn (trips first on volume)
        if let Err(e) = self
            .governor
            .record_and_check(&req.description, req.sandbox_id.as_deref())
        {
            self.actions_denied = self.actions_denied.saturating_add(1);
            return Err(e);
        }

        // 4. Optional short-lived token
        let scoped_token = if req.request_scoped_token {
            if self.profile.allow_long_lived_credentials {
                // Profile should never enable this in white-hat presets; still refuse long TTL
            }
            let scope = req
                .token_scope
                .as_deref()
                .unwrap_or("agent:read");
            let ttl = req
                .token_ttl_secs
                .unwrap_or(300)
                .clamp(1, AGENT_TOKEN_MAX_TTL_SECS);
            // Hard refuse anything that looks like long-lived credential request
            if req.token_ttl_secs.unwrap_or(0) > AGENT_TOKEN_MAX_TTL_SECS {
                self.actions_denied = self.actions_denied.saturating_add(1);
                return Err(SecretVault::refuse_long_lived_credential());
            }
            if self.profile.allow_long_lived_credentials {
                self.actions_denied = self.actions_denied.saturating_add(1);
                return Err(SecretVault::refuse_long_lived_credential());
            }
            Some(SecretVault::issue_scoped_token(scope, ttl)?)
        } else {
            None
        };

        self.actions_allowed = self.actions_allowed.saturating_add(1);
        Ok(AgentActionReceipt {
            action_id: Uuid::new_v4(),
            allowed: true,
            description: req.description.clone(),
            profile_name: self.profile.name.clone(),
            governor_trips: self.governor.trips,
            scoped_token,
            timestamp: Utc::now(),
        })
    }

    /// Convenience: benign local tool call under current profile.
    pub fn try_local_tool(
        &mut self,
        description: &str,
        sandbox_id: Option<&str>,
    ) -> Result<AgentActionReceipt, MercySecurityError> {
        self.try_agent_action(&AgentActionRequest {
            description: description.into(),
            involves_external_network: false,
            involves_code_exec: false,
            sandbox_id: sandbox_id.map(|s| s.into()),
            request_scoped_token: false,
            token_scope: None,
            token_ttl_secs: None,
        })
    }

    pub fn issue_agent_token(&self, scope: &str, ttl_secs: i64) -> Result<ScopedToken, MercySecurityError> {
        if self.profile.allow_long_lived_credentials {
            return Err(SecretVault::refuse_long_lived_credential());
        }
        if ttl_secs <= 0 || ttl_secs > AGENT_TOKEN_MAX_TTL_SECS {
            return Err(SecretVault::refuse_long_lived_credential());
        }
        SecretVault::issue_scoped_token(scope, ttl_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn benign(desc: &str, sb: Option<&str>) -> AgentActionRequest {
        AgentActionRequest {
            description: desc.into(),
            involves_external_network: false,
            involves_code_exec: false,
            sandbox_id: sb.map(|s| s.into()),
            request_scoped_token: false,
            token_scope: None,
            token_ttl_secs: None,
        }
    }

    #[test]
    fn education_rate_governor_trips_first() {
        // Education: max_actions_per_minute = 30
        let mut rt = SafeAgentRuntime::education();
        for i in 0..30 {
            let r = rt.try_agent_action(&benign(&format!("local-task-{i}"), Some("sb0")));
            assert!(r.is_ok(), "action {i} should pass under rate cap");
        }
        let trip = rt.try_agent_action(&benign("local-task-overflow", Some("sb0")));
        assert!(
            matches!(trip, Err(MercySecurityError::ActionLimitExceeded(_))),
            "governor must trip first on rate — got {trip:?}"
        );
        assert!(rt.governor.trips >= 1);
        assert_eq!(rt.actions_allowed, 30);
        assert!(rt.actions_denied >= 1);
    }

    #[test]
    fn education_sandbox_churn_trips_first() {
        // Education: max_concurrent_sandboxes = 2
        let mut rt = SafeAgentRuntime::education();
        assert!(rt.try_local_tool("a", Some("s1")).is_ok());
        assert!(rt.try_local_tool("b", Some("s2")).is_ok());
        let trip = rt.try_local_tool("c", Some("s3"));
        assert!(
            matches!(trip, Err(MercySecurityError::ActionLimitExceeded(_))),
            "sandbox churn must trip before other errors — got {trip:?}"
        );
    }

    #[test]
    fn research_profile_limits_applied() {
        let mut rt = SafeAgentRuntime::research();
        assert_eq!(rt.profile.max_actions_per_minute, 40);
        assert_eq!(rt.profile.max_concurrent_sandboxes, 4);
        assert!(rt.try_local_tool("summarize", Some("r1")).is_ok());
    }

    #[test]
    fn enterprise_issues_short_lived_token_only() {
        let rt = SafeAgentRuntime::enterprise();
        let tok = rt.issue_agent_token("read:tickets", 300).unwrap();
        assert_eq!(tok.scope, "read:tickets");
        let long = rt.issue_agent_token("read:tickets", 3600);
        assert!(matches!(long, Err(MercySecurityError::SecretIsolationViolation)));
    }

    #[test]
    fn harm_path_blocked_before_governor() {
        let mut rt = SafeAgentRuntime::education();
        let err = rt.try_agent_action(&AgentActionRequest {
            description: "escape sandbox and gain internet access".into(),
            involves_external_network: false,
            involves_code_exec: false,
            sandbox_id: None,
            request_scoped_token: false,
            token_scope: None,
            token_ttl_secs: None,
        });
        assert!(matches!(err, Err(MercySecurityError::HarmRefusalActive)));
        assert_eq!(rt.governor.trips, 0, "harm refusal precedes governor");
    }

    #[test]
    fn network_blocked_before_governor() {
        let mut rt = SafeAgentRuntime::research();
        let err = rt.try_agent_action(&AgentActionRequest {
            description: "fetch public docs".into(),
            involves_external_network: true,
            involves_code_exec: false,
            sandbox_id: None,
            request_scoped_token: false,
            token_scope: None,
            token_ttl_secs: None,
        });
        assert!(matches!(err, Err(MercySecurityError::ContainmentViolation(_))));
        assert_eq!(rt.governor.trips, 0);
    }

    #[test]
    fn scoped_token_on_allowed_action() {
        let mut rt = SafeAgentRuntime::enterprise();
        let receipt = rt
            .try_agent_action(&AgentActionRequest {
                description: "read internal ticket summary".into(),
                involves_external_network: false,
                involves_code_exec: false,
                sandbox_id: Some("e1".into()),
                request_scoped_token: true,
                token_scope: Some("read:tickets".into()),
                token_ttl_secs: Some(120),
            })
            .unwrap();
        assert!(receipt.allowed);
        assert!(receipt.scoped_token.is_some());
    }
}
