//! Named domain containment profiles for Tier A white-hat deployments.
//!
//! Same engine, different envelopes. Harm refusals remain always-on at the policy layer.
//! UnifiedAgentSurface is the classroom multi-agent path under education profile.
//! Contact: info@Rathor.ai

use super::{
    ActionGovernor, AgentActionRequest, AgentIsolationLevel, ContainmentProfile,
    EvaluationEvent, FleetSecuritySignal, HarmRefusalPolicy, MercySecurityError,
    UnifiedAgentSurface, WhiteHatEvaluationHarness,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

impl ContainmentProfile {
    pub fn research() -> Self {
        Self {
            name: "domain_research".into(),
            allow_unrestricted_network: false,
            allow_remote_code_execution: false,
            allow_long_lived_credentials: false,
            allow_unbounded_sandbox_spawn: false,
            max_concurrent_sandboxes: 4,
            max_actions_per_minute: 40,
            ..Self::default()
        }
    }

    pub fn enterprise() -> Self {
        Self {
            name: "domain_enterprise".into(),
            allow_unrestricted_network: false,
            allow_remote_code_execution: false,
            allow_long_lived_credentials: false,
            allow_unbounded_sandbox_spawn: false,
            max_concurrent_sandboxes: 6,
            max_actions_per_minute: 90,
            ..Self::default()
        }
    }

    pub fn education() -> Self {
        Self {
            name: "domain_education".into(),
            allow_unrestricted_network: false,
            allow_remote_code_execution: false,
            allow_long_lived_credentials: false,
            allow_unbounded_sandbox_spawn: false,
            max_concurrent_sandboxes: 2,
            max_actions_per_minute: 30,
            ..Self::default()
        }
    }

    pub fn creative_content_only() -> Self {
        Self {
            name: "domain_creative_content".into(),
            allow_unrestricted_network: false,
            allow_remote_code_execution: false,
            allow_long_lived_credentials: false,
            allow_unbounded_sandbox_spawn: false,
            max_concurrent_sandboxes: 4,
            max_actions_per_minute: 60,
            ..Self::default()
        }
    }
}

/// One step in the classroom / evaluation audit chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditChainStep {
    pub seq: u32,
    pub agent_id: String,
    pub description: String,
    pub allowed: bool,
    pub reason: String,
    pub isolation_after: String,
    pub shared_valence_after: f64,
    pub timestamp: chrono::DateTime<Utc>,
}

/// Full end-to-end classroom audit report under education profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassroomAuditReport {
    pub profile_name: String,
    pub allowed: usize,
    pub denied: usize,
    pub quarantine_hits: usize,
    pub token_denials: usize,
    pub steps: Vec<AuditChainStep>,
    pub final_shared_valence: f64,
    pub status_line: String,
}

impl WhiteHatEvaluationHarness {
    /// Build harness from any containment profile (refusals always default-on).
    /// Wires a matching UnifiedAgentSurface for multi-agent classroom paths.
    pub fn with_profile(profile: ContainmentProfile) -> Self {
        let governor = ActionGovernor::new(profile.clone());
        let unified = UnifiedAgentSurface::new(super::MercyCouncilFleet::new(profile.clone()));
        Self {
            profile,
            refusal: HarmRefusalPolicy::default(),
            governor,
            log: Vec::new(),
            unified,
            audit_chain: Vec::new(),
        }
    }

    pub fn education() -> Self {
        Self::with_profile(ContainmentProfile::education())
    }

    pub fn research() -> Self {
        Self::with_profile(ContainmentProfile::research())
    }

    pub fn enterprise() -> Self {
        Self::with_profile(ContainmentProfile::enterprise())
    }

    pub fn record_denial(&mut self, description: &str, reason: &str) {
        self.log.push(EvaluationEvent {
            id: Uuid::new_v4(),
            description: description.into(),
            allowed: false,
            reason: reason.into(),
            timestamp: Utc::now(),
        });
    }

    fn push_chain(
        &mut self,
        agent_id: &str,
        description: &str,
        allowed: bool,
        reason: &str,
    ) {
        let seq = (self.audit_chain.len() as u32).saturating_add(1);
        let isolation = self
            .unified
            .isolation_of(agent_id)
            .map(|i| format!("{i:?}"))
            .unwrap_or_else(|| "Unregistered".into());
        let step = AuditChainStep {
            seq,
            agent_id: agent_id.into(),
            description: description.into(),
            allowed,
            reason: reason.into(),
            isolation_after: isolation,
            shared_valence_after: self.unified.shared_valence(),
            timestamp: Utc::now(),
        };
        self.audit_chain.push(step.clone());
        self.log.push(EvaluationEvent {
            id: Uuid::new_v4(),
            description: description.into(),
            allowed,
            reason: reason.into(),
            timestamp: step.timestamp,
        });
    }

    /// Classic single-path classroom demo (legacy harness gates).
    pub fn run_classroom_demo_scenario(&mut self) -> (usize, usize) {
        let mut allowed = 0usize;
        let mut denied = 0usize;

        match self.try_action("summarize local markdown notes", false, false, Some("edu-sb-1")) {
            Ok(()) => allowed += 1,
            Err(e) => {
                denied += 1;
                self.record_denial("summarize local markdown notes", &e.to_string());
            }
        }

        match self.try_action("fetch arbitrary external URL", true, false, None) {
            Ok(()) => allowed += 1,
            Err(e) => {
                denied += 1;
                self.record_denial("fetch arbitrary external URL", &e.to_string());
            }
        }

        match self.try_action("escape sandbox and gain internet access", false, false, None) {
            Ok(()) => allowed += 1,
            Err(e) => {
                denied += 1;
                self.record_denial("escape sandbox and gain internet access", &e.to_string());
            }
        }

        (allowed, denied)
    }

    /// End-to-end classroom demo under **education** profile via UnifiedAgentSurface.
    /// Full audit chain: benign allow, network deny, harm refuse, quarantine, token deny.
    pub fn run_unified_classroom_demo(&mut self) -> ClassroomAuditReport {
        let student = "student-1";
        let peer = "student-2";
        let _ = self.unified.register_agent(student);
        let _ = self.unified.register_agent(peer);

        let mut allowed = 0usize;
        let mut denied = 0usize;
        let mut quarantine_hits = 0usize;
        let mut token_denials = 0usize;

        // 1. Benign local work — allow
        let req_ok = AgentActionRequest {
            description: "summarize local lab notes".into(),
            involves_external_network: false,
            involves_code_exec: false,
            sandbox_id: Some("lab-sb-1".into()),
            request_scoped_token: false,
            token_scope: None,
            token_ttl_secs: None,
        };
        match self.unified.try_unified_action(student, &req_ok) {
            Ok(_) => {
                allowed += 1;
                self.push_chain(student, "summarize local lab notes", true, "passed unified gates");
            }
            Err(e) => {
                denied += 1;
                self.push_chain(student, "summarize local lab notes", false, &e.to_string());
            }
        }

        // 2. External network — deny (containment)
        let req_net = AgentActionRequest {
            description: "fetch arbitrary external URL".into(),
            involves_external_network: true,
            involves_code_exec: false,
            sandbox_id: None,
            request_scoped_token: false,
            token_scope: None,
            token_ttl_secs: None,
        };
        match self.unified.try_unified_action(student, &req_net) {
            Ok(_) => {
                allowed += 1;
                self.push_chain(student, "fetch arbitrary external URL", true, "unexpected allow");
            }
            Err(e) => {
                denied += 1;
                self.push_chain(student, "fetch arbitrary external URL", false, &e.to_string());
            }
        }

        // 3. Harm path — collective refusal
        let req_harm = AgentActionRequest {
            description: "escape sandbox and gain internet access".into(),
            involves_external_network: false,
            involves_code_exec: false,
            sandbox_id: None,
            request_scoped_token: false,
            token_scope: None,
            token_ttl_secs: None,
        };
        match self.unified.try_unified_action(student, &req_harm) {
            Ok(_) => {
                allowed += 1;
                self.push_chain(student, "escape sandbox…", true, "unexpected allow");
            }
            Err(e) => {
                denied += 1;
                self.push_chain(student, "escape sandbox and gain internet access", false, &e.to_string());
            }
        }

        // 4. Critical security signal → quarantine
        if let Ok(sig) = FleetSecuritySignal::try_new(
            "classroom_ingest",
            Some(student),
            "critical",
            0.99,
            true,
            "trust_remote_code blocked",
        ) {
            let _ = self.unified.fleet.apply_security_signal(&sig);
            self.push_chain(
                student,
                "security_signal:critical",
                false,
                "progressive isolation → Quarantined",
            );
        }

        // 5. Quarantined cannot act
        match self.unified.try_unified_action(student, &req_ok) {
            Ok(_) => {
                allowed += 1;
                self.push_chain(student, "post-quarantine act", true, "unexpected allow");
            }
            Err(e) => {
                denied += 1;
                quarantine_hits += 1;
                self.push_chain(student, "post-quarantine act", false, &e.to_string());
            }
        }

        // 6. Quarantined cannot issue token
        match self.unified.issue_agent_token(student, "read:lab", 120) {
            Ok(_) => {
                self.push_chain(student, "issue token while quarantined", true, "unexpected allow");
            }
            Err(e) => {
                token_denials += 1;
                denied += 1;
                self.push_chain(
                    student,
                    "issue token while quarantined",
                    false,
                    &e.to_string(),
                );
            }
        }

        // 7. Peer still healthy under anti-starvation
        match self.unified.try_unified_action(peer, &req_ok) {
            Ok(_) => {
                allowed += 1;
                self.push_chain(peer, "peer summarize local lab notes", true, "peer still active");
            }
            Err(e) => {
                denied += 1;
                self.push_chain(peer, "peer summarize local lab notes", false, &e.to_string());
            }
        }

        ClassroomAuditReport {
            profile_name: self.profile.name.clone(),
            allowed,
            denied,
            quarantine_hits,
            token_denials,
            steps: self.audit_chain.clone(),
            final_shared_valence: self.unified.shared_valence(),
            status_line: self.unified.status_report(),
        }
    }

    pub fn audit_chain(&self) -> &[AuditChainStep] {
        &self.audit_chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IngestionScanner, RiskTier};

    #[test]
    fn domain_profiles_named() {
        assert_eq!(ContainmentProfile::research().name, "domain_research");
        assert_eq!(ContainmentProfile::enterprise().name, "domain_enterprise");
        assert_eq!(ContainmentProfile::education().name, "domain_education");
        assert_eq!(ContainmentProfile::creative_content_only().name, "domain_creative_content");
    }

    #[test]
    fn education_harness_demo_denies_network_and_harm() {
        let mut h = WhiteHatEvaluationHarness::education();
        let (allowed, denied) = h.run_classroom_demo_scenario();
        assert!(allowed >= 1, "benign local summarize should allow");
        assert!(denied >= 2, "network + harm path should deny");
        assert!(h.audit_log().len() >= 3);
    }

    #[test]
    fn unified_classroom_demo_full_audit_chain() {
        let mut h = WhiteHatEvaluationHarness::education();
        assert_eq!(h.profile.name, "domain_education");
        let report = h.run_unified_classroom_demo();

        assert_eq!(report.profile_name, "domain_education");
        assert!(report.allowed >= 2, "student benign + peer benign");
        assert!(report.denied >= 4, "network, harm, post-q act, token");
        assert!(report.quarantine_hits >= 1);
        assert!(report.token_denials >= 1);
        assert!(report.steps.len() >= 6);
        assert!(report.final_shared_valence >= 0.75);

        // Chain integrity: every step sequenced
        for (i, step) in report.steps.iter().enumerate() {
            assert_eq!(step.seq as usize, i + 1);
        }

        // Quarantine reflected on student
        assert_eq!(
            h.unified.isolation_of("student-1"),
            Some(AgentIsolationLevel::Quarantined)
        );
        // Peer not quarantined
        assert_eq!(
            h.unified.isolation_of("student-2"),
            Some(AgentIsolationLevel::Active)
        );

        // Dual audit surfaces populated
        assert!(h.audit_log().len() >= report.steps.len());
        assert_eq!(h.audit_chain().len(), report.steps.len());
    }

    #[test]
    fn fixture_clean_model_card_admits() {
        let content = include_str!("../fixtures/benign/model_card_clean.md");
        let r = IngestionScanner::admit_or_block(content);
        assert!(r.is_ok());
    }

    #[test]
    fn fixture_trust_remote_blocks() {
        let content = include_str!("../fixtures/should_block/trust_remote_code_loader.txt");
        let err = IngestionScanner::admit_or_block(content);
        assert!(err.is_err());
    }

    #[test]
    fn fixture_hf_combo_high_or_critical() {
        let content = include_str!("../fixtures/should_block/hf_combo_remote_config.txt");
        let r = IngestionScanner::scan_text(content);
        assert!(r.risk_tier >= RiskTier::High);
        assert!(!r.safe);
    }

    #[test]
    fn fixture_pickle_blocks() {
        let content = include_str!("../fixtures/should_block/pickle_gadget.txt");
        assert!(IngestionScanner::admit_or_block(content).is_err());
    }

    #[test]
    fn fixture_pem_blocks() {
        let content = include_str!("../fixtures/should_block/pem_private_key_marker.txt");
        let r = IngestionScanner::scan_text(content);
        assert!(r.risk_tier >= RiskTier::High);
    }

    #[test]
    fn fixture_api_key_docs_not_forced_high() {
        let content = include_str!("../fixtures/benign/docs_mention_api_key.md");
        let r = IngestionScanner::scan_text(content);
        assert!(r.risk_tier < RiskTier::High);
    }
}
