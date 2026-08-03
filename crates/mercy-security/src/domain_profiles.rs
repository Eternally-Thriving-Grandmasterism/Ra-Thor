//! Named domain containment profiles for Tier A white-hat deployments.
//!
//! Same engine, different envelopes. Harm refusals remain always-on at the policy layer.
//! Contact: info@Rathor.ai

use super::{ActionGovernor, ContainmentProfile, HarmRefusalPolicy, WhiteHatEvaluationHarness};

impl ContainmentProfile {
    /// Hub / dataset admission: no external net, no remote code, no long-lived creds.
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

    /// Internal copilots: allow-list networking assumed at higher layer; still no remote code / long-lived creds.
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

    /// Classroom labs: tightest practical envelope; refusals never disabled.
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

    /// Content tools: still deny code-exec and long-lived secrets; network policy stays restricted by default.
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

impl WhiteHatEvaluationHarness {
    /// Build harness from any containment profile (refusals always default-on).
    pub fn with_profile(profile: ContainmentProfile) -> Self {
        let governor = ActionGovernor::new(profile.clone());
        Self {
            profile,
            refusal: HarmRefusalPolicy::default(),
            governor,
            log: Vec::new(),
        }
    }

    /// Classroom preset.
    pub fn education() -> Self {
        Self::with_profile(ContainmentProfile::education())
    }

    /// Research hub preset.
    pub fn research() -> Self {
        Self::with_profile(ContainmentProfile::research())
    }

    /// Enterprise copilot preset.
    pub fn enterprise() -> Self {
        Self::with_profile(ContainmentProfile::enterprise())
    }

    /// Record a denied attempt in the audit log (for demos / education).
    pub fn record_denial(&mut self, description: &str, reason: &str) {
        self.log.push(super::EvaluationEvent {
            id: uuid::Uuid::new_v4(),
            description: description.into(),
            allowed: false,
            reason: reason.into(),
            timestamp: chrono::Utc::now(),
        });
    }

    /// Run a short classroom scenario: benign allow + network deny + harm refusal.
    /// Returns (allowed_count, denied_count).
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
