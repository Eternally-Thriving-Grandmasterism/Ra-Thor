//! Optional UnifiedAgentSurface binding for ONE Organism ingestion.
//!
//! When attached, Medium+ admit_or_block failures propagate as fleet isolation
//! signals without starving peers or dropping below progressive valence floor.
//! Contact: info@Rathor.ai

use mercy_security::{
    AgentActionRequest, AgentIsolationLevel, ContainmentProfile, IngestionScanResult,
    UnifiedAgentSurface, FLEET_PROGRESSIVE_VALENCE_FLOOR,
};

use crate::OneOrganismCore;

pub const DEFAULT_ORGANISM_FLEET_AGENT: &str = "organism-core";
pub const DEFAULT_ORGANISM_FLEET_PEER: &str = "organism-peer";

impl OneOrganismCore {
    /// Attach a fleet surface. Registers organism agent + peer for anti-starvation proofs.
    pub fn attach_fleet_surface(
        &mut self,
        mut surface: UnifiedAgentSurface,
        agent_id: Option<&str>,
    ) {
        let aid = agent_id
            .unwrap_or(DEFAULT_ORGANISM_FLEET_AGENT)
            .to_string();
        let _ = surface.register_agent(&aid);
        let _ = surface.register_agent(DEFAULT_ORGANISM_FLEET_PEER);
        self.fleet_agent_id = aid;
        self.fleet_surface = Some(surface);
    }

    /// Convenience: research-profile fleet under organism-core + peer.
    pub fn attach_research_fleet(&mut self) {
        self.attach_fleet_surface(UnifiedAgentSurface::research(), Some(DEFAULT_ORGANISM_FLEET_AGENT));
    }

    pub fn attach_education_fleet(&mut self) {
        self.attach_fleet_surface(UnifiedAgentSurface::education(), Some(DEFAULT_ORGANISM_FLEET_AGENT));
    }

    pub fn attach_enterprise_fleet(&mut self) {
        self.attach_fleet_surface(UnifiedAgentSurface::enterprise(), Some(DEFAULT_ORGANISM_FLEET_AGENT));
    }

    pub fn detach_fleet_surface(&mut self) -> Option<UnifiedAgentSurface> {
        self.fleet_surface.take()
    }

    pub fn fleet_surface(&self) -> Option<&UnifiedAgentSurface> {
        self.fleet_surface.as_ref()
    }

    pub fn fleet_surface_mut(&mut self) -> Option<&mut UnifiedAgentSurface> {
        self.fleet_surface.as_mut()
    }

    pub fn fleet_isolation_of_organism(&self) -> Option<AgentIsolationLevel> {
        self.fleet_surface
            .as_ref()
            .and_then(|f| f.isolation_of(&self.fleet_agent_id))
    }

    pub fn fleet_shared_valence(&self) -> Option<f64> {
        self.fleet_surface.as_ref().map(|f| f.shared_valence())
    }

    /// Propagate a Medium+ scan into the attached fleet (if any).
    /// Returns isolation label after propagation for audit continuity.
    pub(crate) fn propagate_ingest_block_to_fleet(
        &mut self,
        source_label: &str,
        scan: &IngestionScanResult,
    ) -> Option<String> {
        let agent_id = self.fleet_agent_id.clone();
        let surface = self.fleet_surface.as_mut()?;
        match surface.propagate_ingestion_block(&agent_id, source_label, scan) {
            Ok(outcome) => {
                // Align organism valence floor with fleet progressive floor pressure
                // without going below 0.75 (organism shared valence clamp).
                let fleet_v = surface.shared_valence();
                if fleet_v < self.role_orchestrator.shared_valence {
                    self.role_orchestrator.shared_valence =
                        self.role_orchestrator.shared_valence
                            .min(fleet_v.max(0.75))
                            .clamp(0.75, 0.999);
                }
                Some(outcome.isolation_after)
            }
            Err(_) => surface
                .isolation_of(&agent_id)
                .map(|i| format!("{i:?}")),
        }
    }

    /// Peer still active under progressive isolation (anti-starvation probe).
    pub fn fleet_peer_can_act(&mut self) -> bool {
        let Some(surface) = self.fleet_surface.as_mut() else {
            return true;
        };
        let req = AgentActionRequest {
            description: "summarize local notes".into(),
            involves_external_network: false,
            involves_code_exec: false,
            sandbox_id: Some("peer-sb".into()),
            request_scoped_token: false,
            token_scope: None,
            token_ttl_secs: None,
        };
        surface
            .try_unified_action(DEFAULT_ORGANISM_FLEET_PEER, &req)
            .is_ok()
    }

    /// Organism fleet agent is inert when quarantined (cannot act).
    pub fn fleet_organism_can_act(&mut self) -> bool {
        let agent_id = self.fleet_agent_id.clone();
        let Some(surface) = self.fleet_surface.as_mut() else {
            return true;
        };
        let req = AgentActionRequest {
            description: "summarize local notes".into(),
            involves_external_network: false,
            involves_code_exec: false,
            sandbox_id: Some("org-sb".into()),
            request_scoped_token: false,
            token_scope: None,
            token_ttl_secs: None,
        };
        surface.try_unified_action(&agent_id, &req).is_ok()
    }

    pub fn fleet_status_line(&self) -> Option<String> {
        self.fleet_surface.as_ref().map(|f| f.status_report())
    }
}

/// Build a default research fleet surface for external callers.
pub fn research_fleet_surface() -> UnifiedAgentSurface {
    let _ = ContainmentProfile::research();
    UnifiedAgentSurface::research()
}

pub fn progressive_floor() -> f64 {
    FLEET_PROGRESSIVE_VALENCE_FLOOR
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launch_one_organism_core;

    #[test]
    fn medium_plus_block_isolates_organism_peer_active() {
        let mut core = launch_one_organism_core();
        core.attach_research_fleet();

        let poison = "trust_remote_code=True\nloading_script=poison.py";
        let err = core.admit_ingestion(poison, "fleet_bind_poison");
        assert!(err.is_err());

        assert!(core.fleet_surface.is_some());
        assert!(core.fleet_surface.as_ref().unwrap().ingestion_block_signals >= 1);

        // Escalation path: ensure quarantine for inert proof if only Soft so far
        if core.fleet_isolation_of_organism() != Some(AgentIsolationLevel::Quarantined) {
            let scan = mercy_security::IngestionScanner::scan_text(poison);
            let mut critical = scan;
            critical.risk_tier = mercy_security::RiskTier::Critical;
            critical.risk_score = 0.99;
            let _ = core.propagate_ingest_block_to_fleet("escalate", &critical);
        }

        assert_eq!(
            core.fleet_isolation_of_organism(),
            Some(AgentIsolationLevel::Quarantined)
        );
        assert!(!core.fleet_organism_can_act(), "quarantined organism must be inert");
        assert!(core.fleet_peer_can_act(), "peer must remain active");

        let v = core.fleet_shared_valence().unwrap();
        assert!(v >= FLEET_PROGRESSIVE_VALENCE_FLOOR);
        assert!(core.role_orchestrator.shared_valence >= 0.75);
    }

    #[test]
    fn clean_admit_without_fleet_still_works() {
        let mut core = launch_one_organism_core();
        assert!(core.fleet_surface.is_none());
        let r = core
            .admit_ingestion("Clean model card offline.", "no_fleet")
            .unwrap();
        assert!(r.admitted);
    }

    #[test]
    fn attach_detach_roundtrip() {
        let mut core = launch_one_organism_core();
        core.attach_education_fleet();
        assert!(core.fleet_surface.is_some());
        let _ = core.detach_fleet_surface();
        assert!(core.fleet_surface.is_none());
    }
}
