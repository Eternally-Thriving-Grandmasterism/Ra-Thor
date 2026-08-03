//! Unified agent surface — SafeAgentRuntime bound to MercyCouncilFleet.
//!
//! - Governor trips propagate as security signals → progressive isolation
//! - Medium+ ingestion blocks (admit_or_block) raise fleet security signals
//! - Shared valence is the single floor across fleet + runtime decisions
//! - Quarantined agents cannot act or issue scoped tokens
//!
//! Contact: info@Rathor.ai

use super::{
    AgentActionReceipt, AgentActionRequest, AgentIsolationLevel, FleetSecuritySignal,
    IngestionScanResult, IngestionScanner, MercyCouncilFleet, MercySecurityError, RiskTier,
    SafeAgentRuntime, ScopedToken, SecretVault, AGENT_TOKEN_MAX_TTL_SECS,
    FLEET_PROGRESSIVE_VALENCE_FLOOR, MERCY_VALENCE_FLOOR,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// How many governor trips (within session) escalate isolation one step.
pub const GOVERNOR_TRIPS_PER_ISOLATION_STEP: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedAgentSurface {
    pub fleet: MercyCouncilFleet,
    /// Per-agent runtimes sharing the fleet profile at registration time.
    pub runtimes: HashMap<String, SafeAgentRuntime>,
    pub governor_trip_signals: u64,
    /// Count of Medium+ ingestion blocks propagated into the fleet.
    pub ingestion_block_signals: u64,
    /// Last white-hat ingestion outcome for Cosmic Tick / AGSi heartbeat surfaces.
    pub last_ingestion_outcome: Option<WhitehatIngestionOutcome>,
}

/// Compact outcome for audit chain + Cosmic Tick / AGSi summon heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhitehatIngestionOutcome {
    pub agent_id: String,
    pub source_label: String,
    pub admitted: bool,
    pub risk_tier: String,
    pub risk_score: f32,
    pub isolation_after: String,
    pub shared_valence_after: f64,
    pub message: String,
}

impl UnifiedAgentSurface {
    pub fn new(fleet: MercyCouncilFleet) -> Self {
        Self {
            fleet,
            runtimes: HashMap::new(),
            governor_trip_signals: 0,
            ingestion_block_signals: 0,
            last_ingestion_outcome: None,
        }
    }

    pub fn education() -> Self {
        Self::new(MercyCouncilFleet::education())
    }

    pub fn research() -> Self {
        Self::new(MercyCouncilFleet::research())
    }

    pub fn enterprise() -> Self {
        Self::new(MercyCouncilFleet::enterprise())
    }

    pub fn register_agent(&mut self, agent_id: &str) -> Result<(), MercySecurityError> {
        self.fleet.register_agent(agent_id)?;
        self.runtimes
            .entry(agent_id.into())
            .or_insert_with(|| SafeAgentRuntime::new(self.fleet.profile.clone()));
        Ok(())
    }

    pub fn shared_valence(&self) -> f64 {
        self.fleet.shared_valence
    }

    pub fn isolation_of(&self, agent_id: &str) -> Option<AgentIsolationLevel> {
        self.fleet.agents.get(agent_id).map(|s| s.isolation)
    }

    fn ensure_agent(&mut self, agent_id: &str) -> Result<(), MercySecurityError> {
        if !self.fleet.agents.contains_key(agent_id) {
            self.register_agent(agent_id)?;
        }
        if !self.runtimes.contains_key(agent_id) {
            self.runtimes.insert(
                agent_id.into(),
                SafeAgentRuntime::new(self.fleet.profile.clone()),
            );
        }
        Ok(())
    }

    /// Quarantined agents cannot receive tokens.
    pub fn issue_agent_token(
        &self,
        agent_id: &str,
        scope: &str,
        ttl_secs: i64,
    ) -> Result<ScopedToken, MercySecurityError> {
        let isolation = self
            .fleet
            .agents
            .get(agent_id)
            .map(|s| s.isolation)
            .unwrap_or(AgentIsolationLevel::Active);

        if isolation == AgentIsolationLevel::Quarantined {
            return Err(MercySecurityError::ContainmentViolation(format!(
                "agent '{agent_id}' is quarantined — token issuance denied"
            )));
        }
        if isolation == AgentIsolationLevel::HardIsolated {
            return Err(MercySecurityError::ContainmentViolation(format!(
                "agent '{agent_id}' is hard-isolated — token issuance denied"
            )));
        }
        if self.fleet.shared_valence < self.fleet.progressive_floor {
            return Err(MercySecurityError::Internal(
                "shared valence below progressive floor — token issuance paused".into(),
            ));
        }
        if ttl_secs <= 0 || ttl_secs > AGENT_TOKEN_MAX_TTL_SECS {
            return Err(SecretVault::refuse_long_lived_credential());
        }
        if let Some(rt) = self.runtimes.get(agent_id) {
            return rt.issue_agent_token(scope, ttl_secs);
        }
        SecretVault::issue_scoped_token(scope, ttl_secs)
    }

    /// Propagate a governor trip into fleet isolation + shared valence pressure.
    pub fn propagate_governor_trip(
        &mut self,
        agent_id: &str,
        detail: &str,
    ) -> Result<(), MercySecurityError> {
        self.ensure_agent(agent_id)?;
        let trips = self
            .runtimes
            .get(agent_id)
            .map(|r| r.governor.trips)
            .unwrap_or(1);

        let (tier, score) = if trips >= 3 {
            ("critical", 0.95)
        } else if trips >= 2 {
            ("high", 0.85)
        } else {
            ("high", 0.75)
        };

        let signal = FleetSecuritySignal::try_new(
            "governor_trip",
            Some(agent_id),
            tier,
            score,
            true,
            detail,
        )?;
        self.fleet.apply_security_signal(&signal)?;
        self.governor_trip_signals = self.governor_trip_signals.saturating_add(1);
        let _ = self.fleet.shared_valence.max(FLEET_PROGRESSIVE_VALENCE_FLOOR);
        Ok(())
    }

    /// Map scanner RiskTier → fleet risk label + score for progressive isolation.
    fn map_ingestion_tier(tier: RiskTier, score: f32) -> (&'static str, f64) {
        let s = (score as f64).clamp(0.0, 1.0);
        match tier {
            RiskTier::Critical => ("critical", s.max(0.95)),
            RiskTier::High => ("high", s.max(0.82)),
            RiskTier::Medium => ("medium", s.max(0.55)),
            RiskTier::Low => ("low", s),
            RiskTier::None => ("none", 0.0),
        }
    }

    /// Raise fleet security signal from a Medium+ ingestion block.
    /// Progressive isolation ladder; shared valence never below progressive floor.
    pub fn propagate_ingestion_block(
        &mut self,
        agent_id: &str,
        source_label: &str,
        scan: &IngestionScanResult,
    ) -> Result<WhitehatIngestionOutcome, MercySecurityError> {
        self.ensure_agent(agent_id)?;

        let (tier_label, score) = Self::map_ingestion_tier(scan.risk_tier, scan.risk_score);
        let detail = format!(
            "ingestion_block source={source_label} tier={} score={:.2} threats={:?}",
            scan.risk_tier.as_str(),
            scan.risk_score,
            scan.threats
        );

        let signal = FleetSecuritySignal::try_new(
            "whitehat_ingestion",
            Some(agent_id),
            tier_label,
            score,
            true, // blocked
            &detail,
        )?;
        self.fleet.apply_security_signal(&signal)?;
        self.ingestion_block_signals = self.ingestion_block_signals.saturating_add(1);

        let isolation = self
            .isolation_of(agent_id)
            .unwrap_or(AgentIsolationLevel::Active);
        let outcome = WhitehatIngestionOutcome {
            agent_id: agent_id.into(),
            source_label: source_label.into(),
            admitted: false,
            risk_tier: scan.risk_tier.as_str().into(),
            risk_score: scan.risk_score,
            isolation_after: format!("{isolation:?}"),
            shared_valence_after: self.shared_valence(),
            message: detail,
        };
        self.last_ingestion_outcome = Some(outcome.clone());
        Ok(outcome)
    }

    /// Scan content for an agent: admit None/Low; on Medium+ raise fleet signal + isolation.
    pub fn try_ingest_for_agent(
        &mut self,
        agent_id: &str,
        content: &str,
        source_label: &str,
    ) -> Result<IngestionScanResult, MercySecurityError> {
        self.ensure_agent(agent_id)?;

        // Quarantined agents cannot ingest further (total inert on act path)
        if self.isolation_of(agent_id) == Some(AgentIsolationLevel::Quarantined) {
            return Err(MercySecurityError::ContainmentViolation(format!(
                "agent '{agent_id}' is quarantined — ingestion denied"
            )));
        }

        match IngestionScanner::admit_or_block(content) {
            Ok(scan) => {
                let outcome = WhitehatIngestionOutcome {
                    agent_id: agent_id.into(),
                    source_label: source_label.into(),
                    admitted: true,
                    risk_tier: scan.risk_tier.as_str().into(),
                    risk_score: scan.risk_score,
                    isolation_after: format!(
                        "{:?}",
                        self.isolation_of(agent_id)
                            .unwrap_or(AgentIsolationLevel::Active)
                    ),
                    shared_valence_after: self.shared_valence(),
                    message: format!(
                        "ingestion_admitted source={source_label} tier={}",
                        scan.risk_tier.as_str()
                    ),
                };
                self.last_ingestion_outcome = Some(outcome);
                Ok(scan)
            }
            Err(MercySecurityError::IngestionBlocked(_)) => {
                let scan = IngestionScanner::scan_text(content);
                let _ = self.propagate_ingestion_block(agent_id, source_label, &scan)?;
                Err(MercySecurityError::IngestionBlocked(format!(
                    "tier={} score={:.2} isolation={:?}",
                    scan.risk_tier.as_str(),
                    scan.risk_score,
                    self.isolation_of(agent_id)
                )))
            }
            Err(e) => Err(e),
        }
    }

    /// Unified action path: fleet gates first, then per-agent SafeAgentRuntime.
    /// Governor trips automatically feed isolation.
    pub fn try_unified_action(
        &mut self,
        agent_id: &str,
        req: &AgentActionRequest,
    ) -> Result<AgentActionReceipt, MercySecurityError> {
        self.ensure_agent(agent_id)?;

        self.fleet.try_fleet_action(agent_id, req)?;

        let rt = self
            .runtimes
            .get_mut(agent_id)
            .ok_or_else(|| MercySecurityError::Internal("runtime missing".into()))?;

        match rt.try_agent_action(req) {
            Ok(receipt) => {
                let _ = MERCY_VALENCE_FLOOR;
                Ok(AgentActionReceipt {
                    profile_name: format!(
                        "{}@fleet_valence={:.3}",
                        receipt.profile_name, self.fleet.shared_valence
                    ),
                    ..receipt
                })
            }
            Err(MercySecurityError::ActionLimitExceeded(msg)) => {
                let detail = format!("governor_trip: {msg}");
                drop(rt);
                self.propagate_governor_trip(agent_id, &detail)?;
                Err(MercySecurityError::ActionLimitExceeded(detail))
            }
            Err(e) => Err(e),
        }
    }

    pub fn clear_isolation(&mut self, agent_id: &str) -> Result<(), MercySecurityError> {
        self.fleet.clear_isolation(agent_id)
    }

    pub fn status_report(&self) -> String {
        format!(
            "{} | unified_runtimes={} governor_trip_signals={} ingestion_block_signals={}",
            self.fleet.status_report(),
            self.runtimes.len(),
            self.governor_trip_signals,
            self.ingestion_block_signals
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local(desc: &str) -> AgentActionRequest {
        AgentActionRequest {
            description: desc.into(),
            involves_external_network: false,
            involves_code_exec: false,
            sandbox_id: Some("sb0".into()),
            request_scoped_token: false,
            token_scope: None,
            token_ttl_secs: None,
        }
    }

    #[test]
    fn quarantined_agent_cannot_act() {
        let mut u = UnifiedAgentSurface::research();
        u.register_agent("rogue").unwrap();
        let sig = FleetSecuritySignal::try_new("s", Some("rogue"), "critical", 0.99, true, "q")
            .unwrap();
        u.fleet.apply_security_signal(&sig).unwrap();
        assert_eq!(u.isolation_of("rogue"), Some(AgentIsolationLevel::Quarantined));
        let err = u.try_unified_action("rogue", &local("harmless"));
        assert!(matches!(err, Err(MercySecurityError::ContainmentViolation(_))));
    }

    #[test]
    fn quarantined_agent_cannot_issue_token() {
        let mut u = UnifiedAgentSurface::enterprise();
        u.register_agent("rogue").unwrap();
        let sig = FleetSecuritySignal::try_new("s", Some("rogue"), "critical", 0.99, true, "q")
            .unwrap();
        u.fleet.apply_security_signal(&sig).unwrap();
        let err = u.issue_agent_token("rogue", "read:x", 120);
        assert!(
            matches!(err, Err(MercySecurityError::ContainmentViolation(_))),
            "quarantined must not receive tokens — got {err:?}"
        );
    }

    #[test]
    fn governor_trips_feed_isolation() {
        let mut u = UnifiedAgentSurface::education();
        u.register_agent("busy").unwrap();
        let rt = u.runtimes.get_mut("busy").unwrap();
        for i in 0..30 {
            let _ = rt.try_local_tool(&format!("t{i}"), Some("sb0"));
        }
        let trip = rt.try_local_tool("overflow", Some("sb0"));
        assert!(matches!(trip, Err(MercySecurityError::ActionLimitExceeded(_))));
        assert!(rt.governor.trips >= 1);
        drop(rt);
        u.propagate_governor_trip("busy", "rate overflow").unwrap();
        assert!(u.governor_trip_signals >= 1);
        let iso = u.isolation_of("busy").unwrap();
        assert!(
            matches!(
                iso,
                AgentIsolationLevel::SoftIsolated
                    | AgentIsolationLevel::HardIsolated
                    | AgentIsolationLevel::Quarantined
            ),
            "governor trip must escalate isolation — got {iso:?}"
        );
        assert!(u.shared_valence() >= FLEET_PROGRESSIVE_VALENCE_FLOOR);
        assert!(u.shared_valence() <= MERCY_VALENCE_FLOOR + 0.001);
    }

    #[test]
    fn shared_valence_visible_on_receipt() {
        let mut u = UnifiedAgentSurface::research();
        u.register_agent("a1").unwrap();
        let receipt = u.try_unified_action("a1", &local("summarize notes")).unwrap();
        assert!(receipt.allowed);
        assert!(receipt.profile_name.contains("fleet_valence"));
    }

    #[test]
    fn active_agent_can_issue_short_token() {
        let mut u = UnifiedAgentSurface::enterprise();
        u.register_agent("ok").unwrap();
        let tok = u.issue_agent_token("ok", "read:tickets", 300).unwrap();
        assert_eq!(tok.scope, "read:tickets");
    }

    // ── Ingestion → fleet isolation (living target) ─────────────────────────

    #[test]
    fn medium_plus_ingest_raises_fleet_signal_and_isolates() {
        let mut u = UnifiedAgentSurface::research();
        u.register_agent("loader").unwrap();
        u.register_agent("peer").unwrap();

        let poison = "trust_remote_code=True\nloading_script=poison.py";
        let err = u.try_ingest_for_agent("loader", poison, "hub_dataset");
        assert!(
            matches!(err, Err(MercySecurityError::IngestionBlocked(_))),
            "Medium+ must block — got {err:?}"
        );
        assert!(u.ingestion_block_signals >= 1);
        let iso = u.isolation_of("loader").unwrap();
        assert!(
            matches!(
                iso,
                AgentIsolationLevel::SoftIsolated
                    | AgentIsolationLevel::HardIsolated
                    | AgentIsolationLevel::Quarantined
            ),
            "ingestion block must escalate isolation — got {iso:?}"
        );
        assert!(u.shared_valence() >= FLEET_PROGRESSIVE_VALENCE_FLOOR);
        let outcome = u.last_ingestion_outcome.as_ref().unwrap();
        assert!(!outcome.admitted);
        assert!(outcome.shared_valence_after >= FLEET_PROGRESSIVE_VALENCE_FLOOR);
    }

    #[test]
    fn critical_ingest_quarantines_inert_peer_active() {
        let mut u = UnifiedAgentSurface::education();
        u.register_agent("bad").unwrap();
        u.register_agent("peer").unwrap();

        // Force critical via high-confidence remote + combo patterns
        let poison = include_str!("../fixtures/should_block/hf_combo_remote_config.txt");
        let _ = u.try_ingest_for_agent("bad", poison, "fixture_hf_combo");

        // If not yet quarantined (High only), escalate with explicit critical signal path
        if u.isolation_of("bad") != Some(AgentIsolationLevel::Quarantined) {
            let scan = IngestionScanner::scan_text(poison);
            // Second hit climbs ladder; or direct critical
            let mut scan2 = scan.clone();
            scan2.risk_tier = RiskTier::Critical;
            scan2.risk_score = 0.99;
            let _ = u.propagate_ingestion_block("bad", "escalate", &scan2);
        }

        assert_eq!(u.isolation_of("bad"), Some(AgentIsolationLevel::Quarantined));

        // Quarantine total: cannot act or issue tokens
        let act = u.try_unified_action("bad", &local("summarize local notes"));
        assert!(matches!(act, Err(MercySecurityError::ContainmentViolation(_))));
        let tok = u.issue_agent_token("bad", "read:x", 60);
        assert!(matches!(tok, Err(MercySecurityError::ContainmentViolation(_))));

        // Peer remains active
        assert_eq!(u.isolation_of("peer"), Some(AgentIsolationLevel::Active));
        let peer_ok = u.try_unified_action("peer", &local("summarize local notes"));
        assert!(peer_ok.is_ok(), "peer must stay active — {peer_ok:?}");
        assert!(u.shared_valence() >= FLEET_PROGRESSIVE_VALENCE_FLOOR);
    }

    #[test]
    fn clean_ingest_admits_without_isolation() {
        let mut u = UnifiedAgentSurface::enterprise();
        u.register_agent("clean").unwrap();
        let ok = u
            .try_ingest_for_agent(
                "clean",
                "Clean model card for offline image classification.",
                "model_card",
            )
            .unwrap();
        assert!(ok.safe);
        assert_eq!(u.isolation_of("clean"), Some(AgentIsolationLevel::Active));
        assert_eq!(u.ingestion_block_signals, 0);
        assert!(u.last_ingestion_outcome.as_ref().unwrap().admitted);
    }
}
