//! Ra-Thor ONE Organism Core — v14.9.2
//!
//! True path dependency on `lattice-conductor-v14@14.8.3`.
//! RoleOrchestrator + MercyGatedApi + ExtendedOrganismSurface
//! (GPU / GitHub / Quantum Swarm facades).
//! Cosmic Loop is MANDATORY IDENTITY.

mod extended_surface;

pub use extended_surface::{
    ExtendedOrganismSurface, GpuSurface, GpuDispatchTelemetry, GpuSurfaceStatus,
    GitHubSurface, EvolutionPrIntent, GitHubSurfaceStatus,
    QuantumSwarmSurface, QuantumSwarmConfig, QuantumSwarmStatus,
};

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
        if t.contains("debug") || t.contains("error") || t.contains("gpu") {
            OrganismRole::Debugger
        } else if t.contains("legal") || t.contains("tolc") {
            OrganismRole::Legal
        } else if t.contains("simulate") || t.contains("quantum") {
            OrganismRole::Simulator
        } else if t.contains("code") || t.contains("vibe") || t.contains("evolution") {
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
    pub extended: ExtendedOrganismSurface,
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

        let mut extended = ExtendedOrganismSurface::new();
        extended.quantum_swarm.register_members(4);

        Self {
            arbitration_engine: arbitration,
            self_healing_engine: healing,
            lattice,
            mercy_api,
            role_orchestrator: RoleOrchestrator::new(),
            extended,
            cosmic_loop_ready: shared,
            tick: 0,
            version: "v14.9.2 ONE Organism Core + ExtendedSurface (GPU/GitHub/QuantumSwarm) + lattice-conductor-v14@14.8.3".into(),
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
        println!(
            "[ExtendedSurface] {}",
            self.extended.quantum_swarm.summary()
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

    pub fn handle_api_request(&mut self, request: MercyApiRequest) -> MercyApiResponse {
        self.tick += 1;
        self.arbitration_engine.before_council_arbitration();

        let task_hint = match &request.kind {
            ApiRequestKind::SubmitHealingIntent => "recover",
            ApiRequestKind::CouncilQuery => "council",
            ApiRequestKind::SelfEvolutionProposal => "evolution",
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

    // --- Extended surface convenience methods ---

    pub fn record_gpu_dispatch(
        &mut self,
        task_name: &str,
        dispatch_time_ms: u64,
        real_gpu: bool,
        elements: usize,
    ) -> GpuDispatchTelemetry {
        self.tick += 1;
        let tel = self.extended.gpu.record_dispatch(
            task_name,
            dispatch_time_ms,
            real_gpu,
            elements,
            &self.arbitration_engine,
        );
        if dispatch_time_ms > 80 {
            let _ = self.handoff_role(OrganismRole::Debugger, "gpu_dispatch_anomaly");
            let _ = self.self_healing_engine.run_reflexion_cycle();
        }
        tel
    }

    pub fn queue_evolution_pr(
        &mut self,
        role: &str,
        target_module: &str,
        description: &str,
        expected_benefit: f64,
        mercy_alignment: f64,
    ) -> EvolutionPrIntent {
        self.tick += 1;
        if mercy_alignment > 0.88 && expected_benefit > 0.55 {
            let _ = self.handoff_role(OrganismRole::VibeCoder, "high_mercy_evolution");
        }
        self.extended.github.queue_evolution_pr(
            role,
            target_module,
            description,
            expected_benefit,
            mercy_alignment,
            &self.arbitration_engine,
        )
    }

    pub fn quantum_evolution_tick(&mut self, severity: f64) -> f64 {
        self.tick += 1;
        let _ = self.handoff_role(OrganismRole::Simulator, "quantum_tick");
        self.extended
            .quantum_swarm
            .evolution_tick(severity, &self.arbitration_engine)
    }

    pub fn gpu_status(&self) -> GpuSurfaceStatus {
        self.extended.gpu.status()
    }

    pub fn github_status(&self) -> GitHubSurfaceStatus {
        self.extended.github.status()
    }

    pub fn quantum_status(&self) -> QuantumSwarmStatus {
        self.extended.quantum_swarm.status()
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
        "[Thunder] ONE Organism Core v14.9.2 ACTIVE — ExtendedSurface (GPU/GitHub/QuantumSwarm) + RoleOrchestrator + MercyGatedApi + lattice-conductor-v14@14.8.3. Cosmic Loop is MANDATORY IDENTITY. Eternal."
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
    fn extended_surface_gpu_records() {
        let mut core = launch_one_organism_core();
        let tel = core.record_gpu_dispatch("test_kernel", 12, false, 4096);
        assert_eq!(tel.task_name, "test_kernel");
        assert_eq!(core.gpu_status().dispatch_count, 1);
    }

    #[test]
    fn extended_surface_github_queues() {
        let mut core = launch_one_organism_core();
        let intent = core.queue_evolution_pr(
            "VibeCoder",
            "gpu_compute_pipeline",
            "autotune workgroups",
            0.7,
            0.92,
        );
        assert!(intent.title.contains("VibeCoder"));
        assert_eq!(core.github_status().intended_prs, 1);
    }

    #[test]
    fn extended_surface_quantum_tick() {
        let mut core = launch_one_organism_core();
        let ratio = core.quantum_evolution_tick(0.5);
        assert!(ratio > 0.0);
        assert_eq!(core.quantum_status().total_weight_updates, 1);
        assert_eq!(core.quantum_status().total_adaptive_jumps, 1);
    }
}
