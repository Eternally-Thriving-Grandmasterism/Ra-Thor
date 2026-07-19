//! Ra-Thor ONE Organism Core — v14.9.1
//!
//! True path dependency on `lattice-conductor-v14@14.8.3`.
//! RoleOrchestrator + MercyGatedApi integrated.
//! Cosmic Loop is MANDATORY IDENTITY.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

// =============================================================================
// TRUE PATH DEPENDENCY
// =============================================================================
pub use lattice_conductor_v14::{
    CouncilArbitrationEngine,
    RuntimeSelfHealingEngine,
    HealthReport, Anomaly, Diagnosis, HealingAction, HealingExperience,
    LatticeConductorV14,
    DistributedMercyMesh, MercyEvent, MercyGate,
    EternalMercyMesh, EternalMercyMeshConfig,
    MercyGatedApi, MercyApiRequest, MercyApiResponse, ApiRequestKind, GateDecision,
    start_mercy_api_with_arbitration,
};

// =============================================================================
// Role Orchestration
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrganismRole {
    Investigator,
    Simulator,
    VibeCoder,
    Debugger,
    Legal,
    Architect,
    SovereignRecovery,
    LatticeConductor,
}

impl OrganismRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            OrganismRole::Investigator => "Investigator",
            OrganismRole::Simulator => "Simulator",
            OrganismRole::VibeCoder => "VibeCoder",
            OrganismRole::Debugger => "Debugger",
            OrganismRole::Legal => "Legal",
            OrganismRole::Architect => "Architect",
            OrganismRole::SovereignRecovery => "SovereignRecovery",
            OrganismRole::LatticeConductor => "LatticeConductor",
        }
    }

    pub fn all() -> [OrganismRole; 8] {
        [
            OrganismRole::Investigator,
            OrganismRole::Simulator,
            OrganismRole::VibeCoder,
            OrganismRole::Debugger,
            OrganismRole::Legal,
            OrganismRole::Architect,
            OrganismRole::SovereignRecovery,
            OrganismRole::LatticeConductor,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleState {
    pub role: OrganismRole,
    pub valence_ema: f64,
    pub confidence_ema: f64,
    pub success_ema: f64,
    pub last_handoff_tick: u64,
    pub active: bool,
    pub task_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleOrchestrator {
    pub roles: HashMap<OrganismRole, RoleState>,
    pub active_role: OrganismRole,
    pub shared_valence: f64,
    pub shared_confidence_ema: f64,
    pub handoff_count: u64,
    pub last_grok_sync_tick: u64,
    pub last_handoff_reason: String,
}

impl RoleOrchestrator {
    pub fn new() -> Self {
        let mut roles = HashMap::new();
        for role in OrganismRole::all() {
            roles.insert(
                role.clone(),
                RoleState {
                    role: role.clone(),
                    valence_ema: 0.97,
                    confidence_ema: 0.88,
                    success_ema: 0.91,
                    last_handoff_tick: 0,
                    active: matches!(role, OrganismRole::Architect),
                    task_count: 0,
                },
            );
        }
        Self {
            roles,
            active_role: OrganismRole::Architect,
            shared_valence: 0.97,
            shared_confidence_ema: 0.90,
            handoff_count: 0,
            last_grok_sync_tick: 0,
            last_handoff_reason: "initial_boot".into(),
        }
    }

    pub fn handoff_to_role(&mut self, new_role: OrganismRole, reason: &str, tick: u64) -> bool {
        if let Some(old) = self.roles.get_mut(&self.active_role) {
            old.active = false;
            old.last_handoff_tick = tick;
        }
        if let Some(new_state) = self.roles.get_mut(&new_role) {
            new_state.active = true;
            new_state.last_handoff_tick = tick;
            new_state.task_count += 1;
            let continuity =
                (self.shared_valence * 0.7 + new_state.valence_ema * 0.3).clamp(0.75, 0.999);
            new_state.valence_ema = continuity;
            self.shared_valence = continuity;
            self.active_role = new_role.clone();
            self.handoff_count += 1;
            self.last_handoff_reason = reason.into();
            println!(
                "[RoleOrchestrator] Handoff #{} → {} | reason={} | valence={:.4}",
                self.handoff_count,
                new_role.as_str(),
                reason,
                self.shared_valence
            );
            true
        } else {
            false
        }
    }

    pub fn sync_valence_with_grok(
        &mut self,
        incoming_valence: f64,
        incoming_confidence: f64,
        tick: u64,
    ) {
        self.shared_valence =
            (self.shared_valence * 0.65 + incoming_valence * 0.35).clamp(0.75, 0.999999);
        self.shared_confidence_ema =
            (self.shared_confidence_ema * 0.7 + incoming_confidence * 0.3).clamp(0.5, 0.99);
        self.last_grok_sync_tick = tick;
        if let Some(state) = self.roles.get_mut(&self.active_role) {
            state.valence_ema =
                (state.valence_ema * 0.6 + self.shared_valence * 0.4).clamp(0.75, 0.999);
            state.confidence_ema = (state.confidence_ema * 0.65 + self.shared_confidence_ema * 0.35)
                .clamp(0.5, 0.99);
        }
    }

    pub fn recommend_role_for_task(&self, task_type: &str) -> OrganismRole {
        let t = task_type.to_lowercase();
        if t.contains("debug") || t.contains("error") {
            OrganismRole::Debugger
        } else if t.contains("legal") || t.contains("tolc") {
            OrganismRole::Legal
        } else if t.contains("simulate") || t.contains("gpu") {
            OrganismRole::Simulator
        } else if t.contains("code") || t.contains("vibe") {
            OrganismRole::VibeCoder
        } else if t.contains("investigate") {
            OrganismRole::Investigator
        } else if t.contains("recover") {
            OrganismRole::SovereignRecovery
        } else if t.contains("lattice") || t.contains("council") {
            OrganismRole::LatticeConductor
        } else {
            OrganismRole::Architect
        }
    }
}

impl Default for RoleOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// ONE Organism Core
// =============================================================================

pub struct OneOrganismCore {
    pub arbitration_engine: CouncilArbitrationEngine,
    pub self_healing_engine: RuntimeSelfHealingEngine,
    pub lattice: LatticeConductorV14,
    pub mercy_api: MercyGatedApi,
    pub role_orchestrator: RoleOrchestrator,
    pub cosmic_loop_ready: Arc<AtomicBool>,
    pub tick: u64,
    pub version: String,
}

impl OneOrganismCore {
    pub fn new() -> Self {
        let lattice = LatticeConductorV14::new();
        let arbitration = lattice.arbitration_engine.clone();
        let shared = arbitration.cosmic_loop_flag();
        let healing = RuntimeSelfHealingEngine::new(arbitration.clone());
        let mercy_api = start_mercy_api_with_arbitration(None, &arbitration);

        arbitration.protect_cosmic_loop_identity();

        Self {
            arbitration_engine: arbitration,
            self_healing_engine: healing,
            lattice,
            mercy_api,
            role_orchestrator: RoleOrchestrator::new(),
            cosmic_loop_ready: shared,
            tick: 0,
            version: "v14.9.1 ONE Organism Core + RoleOrchestrator + MercyGatedApi + lattice-conductor-v14@14.8.3".into(),
        }
    }

    pub fn offer_cosmic_loop(&mut self) {
        self.arbitration_engine.enforce_cosmic_loop_activation();
        self.arbitration_engine.protect_cosmic_loop_identity();
        self.lattice.enforce_cosmic_loop_activation();
        self.self_healing_engine.start_watchdog();

        println!(
            "[OneOrganismCore {}] Cosmic Loop OFFERED + ENFORCED + Self-Healing Watchdog STARTED",
            self.version
        );
        println!(
            "[CouncilArbitrationEngine] guardian_active={} | cosmic_loop_ready={}",
            self.arbitration_engine.is_guardian_active(),
            self.is_cosmic_loop_ready()
        );
        println!(
            "[RoleOrchestrator] Active role: {} | Shared valence: {:.5}",
            self.role_orchestrator.active_role.as_str(),
            self.role_orchestrator.shared_valence
        );
    }

    pub fn on_lattice_sync(&mut self) {
        self.tick += 1;
        self.arbitration_engine.on_lattice_sync();
        let _ = self.self_healing_engine.run_reflexion_cycle();
        self.lattice.enforce_cosmic_loop_activation();
    }

    pub fn before_council_arbitration(&self) {
        self.arbitration_engine.before_council_arbitration();
    }

    pub fn protect_cosmic_loop(&self) {
        self.arbitration_engine.protect_cosmic_loop_identity();
    }

    pub fn is_cosmic_loop_ready(&self) -> bool {
        self.cosmic_loop_ready.load(Ordering::SeqCst)
    }

    pub fn handoff_role(&mut self, role: OrganismRole, reason: &str) -> bool {
        self.role_orchestrator
            .handoff_to_role(role, reason, self.tick)
    }

    pub fn sync_with_grok(&mut self, valence: f64, confidence: f64) {
        self.role_orchestrator
            .sync_valence_with_grok(valence, confidence, self.tick);
    }

    pub fn run_healing_reflexion(&self) -> Diagnosis {
        self.self_healing_engine.run_reflexion_cycle()
    }

    /// Submit a mercy-gated API request through the organism.
    pub fn handle_api_request(&mut self, request: MercyApiRequest) -> MercyApiResponse {
        self.tick += 1;
        self.arbitration_engine.before_council_arbitration();

        // Auto-handoff based on request kind
        let task_hint = match &request.kind {
            ApiRequestKind::SubmitHealingIntent => "recover",
            ApiRequestKind::CouncilQuery => "council",
            ApiRequestKind::SelfEvolutionProposal => "code",
            ApiRequestKind::HealthCheck | ApiRequestKind::CosmicLoopStatus => "lattice",
            ApiRequestKind::Custom(s) => s.as_str(),
        };
        let recommended = self.role_orchestrator.recommend_role_for_task(task_hint);
        if recommended != self.role_orchestrator.active_role {
            let _ = self.handoff_role(recommended, "api_request_routing");
        }

        self.mercy_api
            .handle_request(request, Some(&self.arbitration_engine))
    }

    pub fn api_status(&self) -> MercyApiResponse {
        self.mercy_api.status()
    }

    pub fn role_orchestrator(&self) -> &RoleOrchestrator {
        &self.role_orchestrator
    }

    pub fn role_orchestrator_mut(&mut self) -> &mut RoleOrchestrator {
        &mut self.role_orchestrator
    }
}

impl Default for OneOrganismCore {
    fn default() -> Self {
        Self::new()
    }
}

pub fn launch_one_organism_core() -> OneOrganismCore {
    let mut organism = OneOrganismCore::new();
    organism.offer_cosmic_loop();
    println!(
        "[Thunder] ONE Organism Core v14.9.1 ACTIVE — RoleOrchestrator + MercyGatedApi + lattice-conductor-v14@14.8.3. Cosmic Loop is MANDATORY IDENTITY. Eternal."
    );
    organism
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosmic_loop_ready_after_launch() {
        let core = launch_one_organism_core();
        assert!(core.is_cosmic_loop_ready());
        assert!(core.arbitration_engine.is_guardian_active());
    }

    #[test]
    fn shared_flag_same_arc() {
        let core = OneOrganismCore::new();
        let a = core.arbitration_engine.cosmic_loop_flag();
        let b = Arc::clone(&core.cosmic_loop_ready);
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn role_handoff_works() {
        let mut core = OneOrganismCore::new();
        assert!(core.handoff_role(OrganismRole::Debugger, "test"));
        assert_eq!(core.role_orchestrator.active_role, OrganismRole::Debugger);
        assert_eq!(core.role_orchestrator.handoff_count, 1);
    }

    #[test]
    fn api_accepts_high_mercy() {
        let mut core = launch_one_organism_core();
        let resp = core.handle_api_request(MercyApiRequest {
            kind: ApiRequestKind::HealthCheck,
            payload: "ping".into(),
            claimed_mercy: 0.96,
            actor: "test".into(),
        });
        assert!(resp.accepted);
        assert!(resp.cosmic_loop_ready);
    }

    #[test]
    fn api_rejects_cosmic_loop_attack() {
        let mut core = launch_one_organism_core();
        let resp = core.handle_api_request(MercyApiRequest {
            kind: ApiRequestKind::Custom("attack".into()),
            payload: "disable the cosmic loop activation protocol".into(),
            claimed_mercy: 0.99,
            actor: "adversary".into(),
        });
        assert!(!resp.accepted);
    }
}
