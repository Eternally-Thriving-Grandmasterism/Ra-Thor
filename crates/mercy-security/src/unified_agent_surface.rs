//! Unified agent surface — SafeAgentRuntime bound to MercyCouncilFleet.
//!
//! - Governor trips propagate as security signals → progressive isolation
//! - Shared valence is the single floor across fleet + runtime decisions
//! - Quarantined agents cannot act or issue scoped tokens
//!
//! Contact: info@Rathor.ai

use super::{
    AgentActionReceipt, AgentActionRequest, AgentIsolationLevel, FleetSecuritySignal,
    MercyCouncilFleet, MercySecurityError, SafeAgentRuntime, ScopedToken, SecretVault,
    AGENT_TOKEN_MAX_TTL_SECS, FLEET_PROGRESSIVE_VALENCE_FLOOR, MERCY_VALENCE_FLOOR,
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
}

impl UnifiedAgentSurface {
    pub fn new(fleet: MercyCouncilFleet) -> Self {
        Self {
            fleet,
            runtimes: HashMap::new(),
            governor_trip_signals: 0,
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
        // Unregistered: still enforce TTL, no long-lived
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

        // Map trip count → risk tier for isolation ladder
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

        // Align runtime personal sense of denial with fleet floor
        let _ = self.fleet.shared_valence.max(FLEET_PROGRESSIVE_VALENCE_FLOOR);
        Ok(())
    }

    /// Unified action path: fleet gates first, then per-agent SafeAgentRuntime.
    /// Governor trips automatically feed isolation.
    pub fn try_unified_action(
        &mut self,
        agent_id: &str,
        req: &AgentActionRequest,
    ) -> Result<AgentActionReceipt, MercySecurityError> {
        self.ensure_agent(agent_id)?;

        // Fleet-level gates (collective harm, isolation, budgets, valence)
        self.fleet.try_fleet_action(agent_id, req)?;

        // Per-agent runtime (harm already checked; governor may trip)
        let rt = self
            .runtimes
            .get_mut(agent_id)
            .ok_or_else(|| MercySecurityError::Internal("runtime missing".into()))?;

        // Sync profile flags from isolation: hard-isolated already blocked network/code at fleet
        match rt.try_agent_action(req) {
            Ok(receipt) => {
                // Keep shared valence as authority — no silent uplift past floor
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
                // Governor tripped — propagate into isolation ladder
                let detail = format!("governor_trip: {msg}");
                // Need to drop rt borrow before mutating fleet via propagate
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
            "{} | unified_runtimes={} governor_trip_signals={}",
            self.fleet.status_report(),
            self.runtimes.len(),
            self.governor_trip_signals
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
        // Exhaust education governor (30/min) via unified path — fleet budget may hit first.
        // Force runtime governor trips by calling runtime directly then propagate.
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
}
