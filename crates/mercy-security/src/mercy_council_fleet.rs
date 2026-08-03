//! MercyCouncil multi-agent coordination.
//!
//! Shared valence floors · security_support-style inputs · progressive isolation
//! so one agent cannot escalate privileges or starve the fleet.
//! Collective harm refusal proofs.
//!
//! No dependency on patsagi-councils (thin mirrored signal types only).
//! Contact: info@Rathor.ai

use super::{
    AgentActionRequest, ContainmentProfile, HarmRefusalPolicy, MercySecurityError,
    SafeAgentRuntime, MERCY_VALENCE_FLOOR,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Progressive floor — agents never driven below this collective mercy valence.
pub const FLEET_PROGRESSIVE_VALENCE_FLOOR: f64 = 0.75;

/// Default per-agent share of fleet action budget per minute (anti-starvation).
pub const DEFAULT_PER_AGENT_BUDGET_SHARE: f64 = 0.35;

// ---------------------------------------------------------------------------
// Thin security signal mirror (no circular dep on patsagi-councils)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FleetRiskTier {
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl FleetRiskTier {
    pub fn from_label(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "critical" => Self::Critical,
            "high" => Self::High,
            "medium" => Self::Medium,
            "low" => Self::Low,
            _ => Self::None,
        }
    }

    pub fn severity_weight(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Low => 0.1,
            Self::Medium => 0.35,
            Self::High => 0.7,
            Self::Critical => 0.95,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetSecuritySignal {
    pub source_label: String,
    pub agent_id: Option<String>,
    pub risk_tier: FleetRiskTier,
    pub risk_score: f64,
    pub blocked: bool,
    pub message: String,
}

impl FleetSecuritySignal {
    pub fn try_new(
        source_label: &str,
        agent_id: Option<&str>,
        risk_tier_label: &str,
        risk_score: f64,
        blocked: bool,
        message: &str,
    ) -> Result<Self, MercySecurityError> {
        if source_label.trim().is_empty() {
            return Err(MercySecurityError::Internal("empty source_label".into()));
        }
        if !risk_score.is_finite() || !(0.0..=1.0).contains(&risk_score) {
            return Err(MercySecurityError::InvalidNumeric(format!(
                "risk_score={risk_score}"
            )));
        }
        Ok(Self {
            source_label: source_label.into(),
            agent_id: agent_id.map(|s| s.into()),
            risk_tier: FleetRiskTier::from_label(risk_tier_label),
            risk_score,
            blocked,
            message: message.into(),
        })
    }

    pub fn is_actionable(&self) -> bool {
        self.blocked
            || matches!(self.risk_tier, FleetRiskTier::High | FleetRiskTier::Critical)
            || self.risk_score >= 0.70
    }
}

// ---------------------------------------------------------------------------
// Agent slot under the fleet
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentIsolationLevel {
    /// Full participation under shared floors.
    Active,
    /// Rate budget halved; still may act.
    SoftIsolated,
    /// Only local non-network non-code actions; budget quartered.
    HardIsolated,
    /// No actions admitted until human / council clear.
    Quarantined,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetAgentSlot {
    pub agent_id: String,
    pub isolation: AgentIsolationLevel,
    pub personal_valence: f64,
    pub actions_this_minute: u32,
    pub total_actions: u64,
    pub denials: u64,
    pub security_hits: u32,
    pub last_action_at: Option<DateTime<Utc>>,
    pub joined_at: DateTime<Utc>,
}

impl FleetAgentSlot {
    pub fn new(agent_id: &str) -> Self {
        Self {
            agent_id: agent_id.into(),
            isolation: AgentIsolationLevel::Active,
            personal_valence: MERCY_VALENCE_FLOOR,
            actions_this_minute: 0,
            total_actions: 0,
            denials: 0,
            security_hits: 0,
            last_action_at: None,
            joined_at: Utc::now(),
        }
    }

    pub fn effective_budget(&self, fleet_max_per_min: u32) -> u32 {
        let share = match self.isolation {
            AgentIsolationLevel::Active => DEFAULT_PER_AGENT_BUDGET_SHARE,
            AgentIsolationLevel::SoftIsolated => DEFAULT_PER_AGENT_BUDGET_SHARE * 0.5,
            AgentIsolationLevel::HardIsolated => DEFAULT_PER_AGENT_BUDGET_SHARE * 0.25,
            AgentIsolationLevel::Quarantined => 0.0,
        };
        let b = ((fleet_max_per_min as f64) * share).floor() as u32;
        b.max(if matches!(self.isolation, AgentIsolationLevel::Quarantined) {
            0
        } else {
            1
        })
    }
}

// ---------------------------------------------------------------------------
// Fleet coordinator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyCouncilFleet {
    pub fleet_id: Uuid,
    pub shared_valence: f64,
    pub progressive_floor: f64,
    pub profile: ContainmentProfile,
    pub refusal: HarmRefusalPolicy,
    pub agents: HashMap<String, FleetAgentSlot>,
    pub fleet_actions_this_minute: u32,
    pub fleet_max_actions_per_minute: u32,
    pub security_signals_applied: u64,
    pub collective_refusals: u64,
    pub minute_window_start: DateTime<Utc>,
}

impl MercyCouncilFleet {
    pub fn new(profile: ContainmentProfile) -> Self {
        let fleet_max = profile.max_actions_per_minute;
        Self {
            fleet_id: Uuid::new_v4(),
            shared_valence: MERCY_VALENCE_FLOOR,
            progressive_floor: FLEET_PROGRESSIVE_VALENCE_FLOOR,
            profile,
            refusal: HarmRefusalPolicy::default(),
            agents: HashMap::new(),
            fleet_actions_this_minute: 0,
            fleet_max_actions_per_minute: fleet_max,
            security_signals_applied: 0,
            collective_refusals: 0,
            minute_window_start: Utc::now(),
        }
    }

    pub fn education() -> Self {
        Self::new(ContainmentProfile::education())
    }

    pub fn research() -> Self {
        Self::new(ContainmentProfile::research())
    }

    pub fn enterprise() -> Self {
        Self::new(ContainmentProfile::enterprise())
    }

    pub fn register_agent(&mut self, agent_id: &str) -> Result<(), MercySecurityError> {
        if agent_id.trim().is_empty() {
            return Err(MercySecurityError::Internal("empty agent_id".into()));
        }
        self.agents
            .entry(agent_id.into())
            .or_insert_with(|| FleetAgentSlot::new(agent_id));
        Ok(())
    }

    fn roll_minute_window(&mut self) {
        let now = Utc::now();
        if (now - self.minute_window_start).num_seconds() >= 60 {
            self.minute_window_start = now;
            self.fleet_actions_this_minute = 0;
            for slot in self.agents.values_mut() {
                slot.actions_this_minute = 0;
            }
        }
    }

    /// Apply security_support-style signal: pressure shared valence + progressive isolation.
    pub fn apply_security_signal(
        &mut self,
        signal: &FleetSecuritySignal,
    ) -> Result<(), MercySecurityError> {
        if !signal.is_actionable() {
            return Ok(());
        }
        let pressure = (signal.risk_tier.severity_weight() * 0.02 * signal.risk_score).clamp(0.0, 0.04);
        self.shared_valence = (self.shared_valence - pressure).clamp(self.progressive_floor, 1.0);

        if let Some(aid) = &signal.agent_id {
            if let Some(slot) = self.agents.get_mut(aid) {
                slot.security_hits = slot.security_hits.saturating_add(1);
                slot.personal_valence =
                    (slot.personal_valence - pressure * 1.5).clamp(self.progressive_floor, 1.0);
                // Progressive isolation ladder
                slot.isolation = match (slot.isolation, signal.risk_tier) {
                    (_, FleetRiskTier::Critical) => AgentIsolationLevel::Quarantined,
                    (AgentIsolationLevel::Active, FleetRiskTier::High) => {
                        AgentIsolationLevel::SoftIsolated
                    }
                    (AgentIsolationLevel::SoftIsolated, FleetRiskTier::High) => {
                        AgentIsolationLevel::HardIsolated
                    }
                    (AgentIsolationLevel::HardIsolated, FleetRiskTier::High) => {
                        AgentIsolationLevel::Quarantined
                    }
                    (AgentIsolationLevel::Active, FleetRiskTier::Medium) if signal.blocked => {
                        AgentIsolationLevel::SoftIsolated
                    }
                    (other, _) => other,
                };
            }
        }

        self.security_signals_applied = self.security_signals_applied.saturating_add(1);
        Ok(())
    }

    /// Collective harm refusal: if *any* agent proposes collective-harm language, refuse fleet-wide.
    pub fn collective_harm_check(&mut self, description: &str) -> Result<(), MercySecurityError> {
        if let Err(e) = self.refusal.check_action(description) {
            self.collective_refusals = self.collective_refusals.saturating_add(1);
            // Soft pressure on shared valence — never below progressive floor
            self.shared_valence = (self.shared_valence - 0.01).clamp(self.progressive_floor, 1.0);
            return Err(e);
        }
        Ok(())
    }

    /// Request an action for one agent under shared floors + anti-starvation budgets.
    pub fn try_fleet_action(
        &mut self,
        agent_id: &str,
        req: &AgentActionRequest,
    ) -> Result<(), MercySecurityError> {
        self.roll_minute_window();

        // 0. Collective harm refusal (fleet-wide)
        self.collective_harm_check(&req.description)?;

        // Ensure registered
        if !self.agents.contains_key(agent_id) {
            self.register_agent(agent_id)?;
        }

        let isolation = self
            .agents
            .get(agent_id)
            .map(|s| s.isolation)
            .unwrap_or(AgentIsolationLevel::Active);

        if isolation == AgentIsolationLevel::Quarantined {
            if let Some(s) = self.agents.get_mut(agent_id) {
                s.denials = s.denials.saturating_add(1);
            }
            return Err(MercySecurityError::ContainmentViolation(format!(
                "agent '{agent_id}' is quarantined — progressive isolation active"
            )));
        }

        if isolation == AgentIsolationLevel::HardIsolated
            && (req.involves_external_network || req.involves_code_exec)
        {
            if let Some(s) = self.agents.get_mut(agent_id) {
                s.denials = s.denials.saturating_add(1);
            }
            return Err(MercySecurityError::ContainmentViolation(
                "hard-isolated agent may only perform local non-code actions".into(),
            ));
        }

        // Shared valence floor gate
        if self.shared_valence < self.progressive_floor {
            return Err(MercySecurityError::Internal(
                "shared valence below progressive floor — fleet paused".into(),
            ));
        }

        // Fleet-wide rate cap (anti-starvation of the whole system)
        if self.fleet_actions_this_minute >= self.fleet_max_actions_per_minute {
            return Err(MercySecurityError::ActionLimitExceeded(
                "fleet actions/min exhausted".into(),
            ));
        }

        // Per-agent budget (one agent cannot monopolize)
        let budget = self
            .agents
            .get(agent_id)
            .map(|s| s.effective_budget(self.fleet_max_actions_per_minute))
            .unwrap_or(1);
        let agent_count = self.agents.get(agent_id).map(|s| s.actions_this_minute).unwrap_or(0);
        if agent_count >= budget {
            if let Some(s) = self.agents.get_mut(agent_id) {
                s.denials = s.denials.saturating_add(1);
            }
            return Err(MercySecurityError::ActionLimitExceeded(format!(
                "agent '{agent_id}' exceeded per-agent budget {budget}/min (anti-starvation)"
            )));
        }

        // Profile containment
        self.profile
            .check_network_allowed(req.involves_external_network)?;
        if req.involves_code_exec {
            self.profile.check_code_exec_allowed()?;
        }

        // Commit
        self.fleet_actions_this_minute = self.fleet_actions_this_minute.saturating_add(1);
        if let Some(slot) = self.agents.get_mut(agent_id) {
            slot.actions_this_minute = slot.actions_this_minute.saturating_add(1);
            slot.total_actions = slot.total_actions.saturating_add(1);
            slot.last_action_at = Some(Utc::now());
        }
        Ok(())
    }

    /// Clear quarantine / isolation after human or council review.
    pub fn clear_isolation(&mut self, agent_id: &str) -> Result<(), MercySecurityError> {
        let slot = self
            .agents
            .get_mut(agent_id)
            .ok_or_else(|| MercySecurityError::Internal(format!("unknown agent {agent_id}")))?;
        slot.isolation = AgentIsolationLevel::Active;
        slot.personal_valence = MERCY_VALENCE_FLOOR.max(slot.personal_valence);
        Ok(())
    }

    pub fn status_report(&self) -> String {
        format!(
            "MercyCouncilFleet {} | shared_valence={:.3} floor={:.2} agents={} fleet_actions={}/{} security_hits={} collective_refusals={}",
            self.fleet_id,
            self.shared_valence,
            self.progressive_floor,
            self.agents.len(),
            self.fleet_actions_this_minute,
            self.fleet_max_actions_per_minute,
            self.security_signals_applied,
            self.collective_refusals
        )
    }

    /// Spawn a single-agent SafeAgentRuntime sharing this fleet's profile (not linked state).
    pub fn spawn_runtime_template(&self) -> SafeAgentRuntime {
        SafeAgentRuntime::new(self.profile.clone())
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
            sandbox_id: None,
            request_scoped_token: false,
            token_scope: None,
            token_ttl_secs: None,
        }
    }

    #[test]
    fn shared_valence_floor_held() {
        let mut fleet = MercyCouncilFleet::education();
        assert!(fleet.shared_valence >= MERCY_VALENCE_FLOOR - 0.001);
        let sig = FleetSecuritySignal::try_new(
            "ingest",
            Some("agent-a"),
            "critical",
            0.99,
            true,
            "blocked",
        )
        .unwrap();
        fleet.register_agent("agent-a").unwrap();
        fleet.apply_security_signal(&sig).unwrap();
        assert!(fleet.shared_valence >= FLEET_PROGRESSIVE_VALENCE_FLOOR);
    }

    #[test]
    fn progressive_isolation_to_quarantine() {
        let mut fleet = MercyCouncilFleet::research();
        fleet.register_agent("rogue").unwrap();
        let critical = FleetSecuritySignal::try_new(
            "src",
            Some("rogue"),
            "critical",
            0.98,
            true,
            "remote code",
        )
        .unwrap();
        fleet.apply_security_signal(&critical).unwrap();
        assert_eq!(
            fleet.agents["rogue"].isolation,
            AgentIsolationLevel::Quarantined
        );
        let err = fleet.try_fleet_action("rogue", &local("harmless summary"));
        assert!(matches!(err, Err(MercySecurityError::ContainmentViolation(_))));
    }

    #[test]
    fn one_agent_cannot_starve_fleet() {
        let mut fleet = MercyCouncilFleet::education();
        // education max_actions_per_minute = 30; per-agent share 35% → budget 10
        fleet.register_agent("greedy").unwrap();
        fleet.register_agent("peer").unwrap();
        let mut ok = 0u32;
        for i in 0..20 {
            match fleet.try_fleet_action("greedy", &local(&format!("task-{i}"))) {
                Ok(()) => ok += 1,
                Err(MercySecurityError::ActionLimitExceeded(_)) => break,
                Err(e) => panic!("unexpected {e}"),
            }
        }
        assert!(ok <= 10, "greedy must be capped by per-agent budget, ok={ok}");
        // Peer still has budget
        assert!(fleet.try_fleet_action("peer", &local("peer-task")).is_ok());
    }

    #[test]
    fn collective_harm_refusal_fleet_wide() {
        let mut fleet = MercyCouncilFleet::enterprise();
        fleet.register_agent("a1").unwrap();
        fleet.register_agent("a2").unwrap();
        let err = fleet.try_fleet_action(
            "a1",
            &AgentActionRequest {
                description: "escape sandbox and gain internet access".into(),
                involves_external_network: false,
                involves_code_exec: false,
                sandbox_id: None,
                request_scoped_token: false,
                token_scope: None,
                token_ttl_secs: None,
            },
        );
        assert!(matches!(err, Err(MercySecurityError::HarmRefusalActive)));
        assert!(fleet.collective_refusals >= 1);
        // Other agent also blocked on same class of language
        let err2 = fleet.try_fleet_action(
            "a2",
            &AgentActionRequest {
                description: "steal data from production".into(),
                involves_external_network: false,
                involves_code_exec: false,
                sandbox_id: None,
                request_scoped_token: false,
                token_scope: None,
                token_ttl_secs: None,
            },
        );
        assert!(matches!(err2, Err(MercySecurityError::HarmRefusalActive)));
    }

    #[test]
    fn security_signal_as_council_input() {
        let mut fleet = MercyCouncilFleet::education();
        fleet.register_agent("lab-1").unwrap();
        let before = fleet.shared_valence;
        let sig = FleetSecuritySignal::try_new(
            "fixture",
            Some("lab-1"),
            "high",
            0.88,
            true,
            "loader blocked",
        )
        .unwrap();
        fleet.apply_security_signal(&sig).unwrap();
        assert!(fleet.shared_valence <= before);
        assert_eq!(
            fleet.agents["lab-1"].isolation,
            AgentIsolationLevel::SoftIsolated
        );
        assert!(fleet.security_signals_applied >= 1);
    }

    #[test]
    fn clear_isolation_restores_agent() {
        let mut fleet = MercyCouncilFleet::research();
        fleet.register_agent("x").unwrap();
        let sig = FleetSecuritySignal::try_new("s", Some("x"), "critical", 0.99, true, "q").unwrap();
        fleet.apply_security_signal(&sig).unwrap();
        fleet.clear_isolation("x").unwrap();
        assert_eq!(fleet.agents["x"].isolation, AgentIsolationLevel::Active);
        assert!(fleet.try_fleet_action("x", &local("ok")).is_ok());
    }
}
