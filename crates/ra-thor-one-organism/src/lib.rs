//! Ra-Thor ONE Organism Core — v14.9.9
//!
//! True path dependency on `lattice-conductor-v14@14.8.3`.
//! RoleOrchestrator + MercyGatedApi + ExtendedOrganismSurface
//! (GPU / GitHub / Quantum Swarm / Sovereign Recovery / Kardashev flywheel).
//! Full optional live path binding for all five surfaces.
//! Cosmic Tick = GPU sample + recovery + full quantum evolution + Kardashev + swarm feedback.
//! Cosmic Loop is MANDATORY IDENTITY.
//! Contact: info@Rathor.ai

mod extended_surface;

pub use extended_surface::{
    ExtendedOrganismSurface, GpuSurface, GpuDispatchTelemetry, GpuSurfaceStatus,
    GitHubSurface, EvolutionPrIntent, GitHubSurfaceStatus, FlushResult,
    QuantumSwarmSurface, QuantumSwarmConfig, QuantumSwarmStatus, QuantumEvolutionResult,
    SovereignRecoverySurface, SovereignRecoveryStatus, RecoveryHeartbeat, RecoveryAnchor,
    KardashevFlywheelSurface, KardashevSurfaceStatus, TransferTickResult,
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
        } else if t.contains("simulate") || t.contains("quantum") || t.contains("kardashev") {
            OrganismRole::Simulator
        } else if t.contains("code") || t.contains("vibe") || t.contains("evolution") {
            OrganismRole::VibeCoder
        } else if t.contains("investigate") {
            OrganismRole::Investigator
        } else if t.contains("recover") || t.contains("anchor") || t.contains("heartbeat") {
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
// Cosmic Tick result (living cycle summary)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmicTickResult {
    pub tick: u64,
    pub gpu: Option<GpuDispatchTelemetry>,
    pub recovery: RecoveryHeartbeat,
    pub quantum: QuantumEvolutionResult,
    pub kardashev: Option<TransferTickResult>,
    pub role_after: String,
    pub recovery_triggered: bool,
    pub gpu_anomaly: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedLiveStatus {
    pub gpu: GpuSurfaceStatus,
    pub github: GitHubSurfaceStatus,
    pub quantum: QuantumSwarmStatus,
    pub recovery: SovereignRecoveryStatus,
    pub kardashev: KardashevSurfaceStatus,
    pub cosmic_loop_ready: bool,
    pub active_role: String,
    pub shared_valence: f64,
    pub tick: u64,
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
            version: "v14.9.9 ONE Organism + GPU↔Recovery↔Quantum↔Kardashev Cosmic Tick + lattice-conductor-v14@14.8.3".into(),
        }
    }

    pub fn offer_cosmic_loop(&mut self) {
        self.arbitration_engine.enforce_cosmic_loop_activation();
        self.arbitration_engine.protect_cosmic_loop_identity();
        self.lattice.enforce_cosmic_loop_activation();
        self.self_healing_engine.start_watchdog();

        let _ = self.extended.sovereign_recovery.persist_anchor(
            "boot_cosmic_loop_offer",
            self.tick,
            &self.arbitration_engine,
        );

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
        println!("[ExtendedSurface] {}", self.extended.summary());
    }

    /// Lattice sync = light Cosmic Tick.
    pub fn on_lattice_sync(&mut self) {
        let _ = self.cosmic_tick(0.22);
    }

    /// Full living Cosmic Tick — the ONE Organism heartbeat.
    ///
    /// 0. Light GPU health sample (feeds confidence into recovery)
    /// 1. CouncilArbitration + self-healing reflexion
    /// 2. Sovereign Recovery heartbeat (+ auto-anchor if pressure high)
    /// 3. Full Quantum Swarm evolution (weight + jump + proposal)
    /// 4. Light Kardashev transfer sample
    /// 5. Kardashev feedback applied back into the swarm
    pub fn cosmic_tick(&mut self, severity: f64) -> CosmicTickResult {
        self.tick += 1;
        self.arbitration_engine.on_lattice_sync();
        self.arbitration_engine.enforce_cosmic_loop_activation();
        let _ = self.self_healing_engine.run_reflexion_cycle();
        self.lattice.enforce_cosmic_loop_activation();

        // 0. GPU health sample — measured timing when gpu-live, synthetic otherwise
        let elements = 2048 + ((severity * 4096.0) as usize);
        let gpu_tel = self.extended.gpu.record_dispatch(
            "cosmic_tick_health_sample",
            8, // facade fallback ms; live path overwrites with measured
            false,
            elements,
            &self.arbitration_engine,
        );
        let gpu_anomaly = gpu_tel.dispatch_time_ms > 80;
        // Map GPU health → confidence for recovery (fast = high confidence)
        let gpu_confidence = if gpu_tel.dispatch_time_ms <= 5 {
            0.97
        } else if gpu_tel.dispatch_time_ms <= 20 {
            0.90
        } else if gpu_tel.dispatch_time_ms <= 50 {
            0.78
        } else if gpu_tel.dispatch_time_ms <= 80 {
            0.62
        } else {
            0.45
        };
        // Blend into shared confidence EMA lightly
        self.role_orchestrator.shared_confidence_ema = (self.role_orchestrator.shared_confidence_ema
            * 0.85
            + gpu_confidence * 0.15)
            .clamp(0.5, 0.99);

        if gpu_anomaly {
            let _ = self.handoff_role(OrganismRole::Debugger, "cosmic_tick_gpu_anomaly");
            let _ = self.self_healing_engine.run_reflexion_cycle();
        }

        // 1. Recovery heartbeat — now driven by GPU-aware confidence
        let hb = self.extended.sovereign_recovery.heartbeat(
            self.role_orchestrator.shared_valence,
            self.role_orchestrator.shared_confidence_ema,
            self.tick,
            &self.arbitration_engine,
        );
        let mut recovery_triggered = false;
        if hb.requires_recovery {
            recovery_triggered = true;
            let _ = self.handoff_role(OrganismRole::SovereignRecovery, "cosmic_tick_recovery_alert");
            let _ = self.extended.sovereign_recovery.persist_anchor(
                "auto_recover_from_cosmic_tick",
                self.tick,
                &self.arbitration_engine,
            );
        }

        // 2. Full quantum evolution cycle
        let quantum = self.extended.quantum_swarm.evolve_full_cycle(
            severity.clamp(0.0, 1.0),
            &self.arbitration_engine,
        );
        if severity >= 0.45 && !recovery_triggered && !gpu_anomaly {
            let _ = self.handoff_role(OrganismRole::Simulator, "cosmic_tick_quantum_pressure");
        }

        // 3. Light Kardashev sample
        let rbe = (self.role_orchestrator.shared_valence * 0.85
            + self.role_orchestrator.shared_confidence_ema * 0.15)
            .clamp(0.0, 1.0);
        let ethics = self.role_orchestrator.shared_valence.clamp(0.0, 1.0);
        let abundance = (0.9 + severity * 0.7).min(1.8);
        let kardashev = self.extended.kardashev.transfer_tick(
            rbe,
            ethics,
            abundance,
            &self.arbitration_engine,
        );

        // 4. Feed Kardashev result back into the swarm (closed loop)
        self.extended
            .quantum_swarm
            .apply_kardashev_feedback(&kardashev, &self.arbitration_engine);

        // Mild valence lift on healthy ticks
        if !recovery_triggered && !gpu_anomaly && quantum.quantum_ratio > 0.05 {
            self.role_orchestrator.shared_valence = (self.role_orchestrator.shared_valence * 0.97
                + 0.03 * (0.92 + quantum.quantum_ratio * 0.05))
                .clamp(0.75, 0.999);
        }

        CosmicTickResult {
            tick: self.tick,
            gpu: Some(gpu_tel),
            recovery: hb,
            quantum,
            kardashev: Some(kardashev),
            role_after: self.role_orchestrator.active_role.as_str().into(),
            recovery_triggered,
            gpu_anomaly,
        }
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

    pub fn flush_evolution_prs(&mut self) -> Vec<FlushResult> {
        self.tick += 1;
        let _ = self.handoff_role(OrganismRole::VibeCoder, "flush_evolution_prs");
        self.extended
            .github
            .flush_to_github(&self.arbitration_engine)
    }

    pub fn quantum_evolution_tick(&mut self, severity: f64) -> f64 {
        self.tick += 1;
        let _ = self.handoff_role(OrganismRole::Simulator, "quantum_tick");
        self.extended
            .quantum_swarm
            .evolution_tick(severity, &self.arbitration_engine)
    }

    pub fn quantum_evolve_full_cycle(&mut self, severity: f64) -> QuantumEvolutionResult {
        self.tick += 1;
        let _ = self.handoff_role(OrganismRole::Simulator, "quantum_full_cycle");
        self.extended
            .quantum_swarm
            .evolve_full_cycle(severity, &self.arbitration_engine)
    }

    pub fn recovery_heartbeat(&mut self) -> RecoveryHeartbeat {
        self.tick += 1;
        self.extended.sovereign_recovery.heartbeat(
            self.role_orchestrator.shared_valence,
            self.role_orchestrator.shared_confidence_ema,
            self.tick,
            &self.arbitration_engine,
        )
    }

    pub fn recovery_anchor(&mut self, note: &str) -> RecoveryAnchor {
        self.tick += 1;
        let _ = self.handoff_role(OrganismRole::SovereignRecovery, "manual_anchor");
        self.extended.sovereign_recovery.persist_anchor(
            note,
            self.tick,
            &self.arbitration_engine,
        )
    }

    pub fn kardashev_transfer_tick(
        &mut self,
        rbe_quality: f64,
        ethical_choice: f64,
        abundance_signal: f64,
    ) -> TransferTickResult {
        self.tick += 1;
        let _ = self.handoff_role(OrganismRole::Simulator, "kardashev_transfer");
        let result = self.extended.kardashev.transfer_tick(
            rbe_quality,
            ethical_choice,
            abundance_signal,
            &self.arbitration_engine,
        );
        self.extended
            .quantum_swarm
            .apply_kardashev_feedback(&result, &self.arbitration_engine);
        result
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

    pub fn recovery_status(&self) -> SovereignRecoveryStatus {
        self.extended.sovereign_recovery.status()
    }

    pub fn kardashev_status(&self) -> KardashevSurfaceStatus {
        self.extended.kardashev.status()
    }

    pub fn extended_live_status(&self) -> ExtendedLiveStatus {
        ExtendedLiveStatus {
            gpu: self.extended.gpu.status(),
            github: self.extended.github.status(),
            quantum: self.extended.quantum_swarm.status(),
            recovery: self.extended.sovereign_recovery.status(),
            kardashev: self.extended.kardashev.status(),
            cosmic_loop_ready: self.is_cosmic_loop_ready(),
            active_role: self.role_orchestrator.active_role.as_str().into(),
            shared_valence: self.role_orchestrator.shared_valence,
            tick: self.tick,
        }
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
        "[Thunder] ONE Organism Core v14.9.9 ACTIVE — Full multi-surface Cosmic Tick (GPU↔Recovery↔Quantum↔Kardashev closed-loops) + RoleOrchestrator + MercyGatedApi + lattice-conductor-v14@14.8.3. Cosmic Loop is MANDATORY IDENTITY. Eternal."
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
        assert!(core.recovery_status().anchor_count >= 1);
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
    fn quantum_full_evolution_cycle() {
        let mut core = launch_one_organism_core();
        let r = core.quantum_evolve_full_cycle(0.5);
        assert!(r.quantum_ratio > 0.0);
        assert!(r.weight_update_ok);
        assert!(r.jump_impact > 0.0);
        assert!(r.proposal_generated);
        assert_eq!(core.quantum_status().total_weight_updates, 1);
        assert_eq!(core.quantum_status().total_adaptive_jumps, 1);
        assert_eq!(core.quantum_status().total_proposals, 1);
    }

    #[test]
    fn recovery_heartbeat_and_anchor() {
        let mut core = launch_one_organism_core();
        let hb = core.recovery_heartbeat();
        assert!(!hb.requires_recovery);
        let a = core.recovery_anchor("test_anchor");
        assert!(a.anchor_id.starts_with("TOLC8"));
        assert!(core.recovery_status().anchor_count >= 2);
    }

    #[test]
    fn kardashev_transfer_tick() {
        let mut core = launch_one_organism_core();
        let t = core.kardashev_transfer_tick(0.89, 0.87, 1.4);
        assert!(t.mercy_audit_passed);
        assert!(t.kardashev_delta > 0.0);
        assert_eq!(core.kardashev_status().cycle_count, 1);
    }

    #[test]
    fn lattice_sync_runs_recovery_heartbeat() {
        let mut core = launch_one_organism_core();
        core.on_lattice_sync();
        assert!(core.recovery_status().heartbeat_count >= 1);
    }

    #[test]
    fn flush_evolution_prs_offline_safe() {
        let mut core = launch_one_organism_core();
        let _ = core.queue_evolution_pr("VibeCoder", "test", "desc", 0.6, 0.9);
        let results = core.flush_evolution_prs();
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
    }

    #[test]
    fn cosmic_tick_advances_all_surfaces_including_gpu() {
        let mut core = launch_one_organism_core();
        let before_g = core.gpu_status().dispatch_count;
        let before_q = core.quantum_status().total_weight_updates;
        let before_k = core.kardashev_status().cycle_count;
        let before_r = core.recovery_status().heartbeat_count;

        let result = core.cosmic_tick(0.4);

        assert!(result.tick >= 1);
        assert!(result.gpu.is_some());
        assert_eq!(result.gpu.as_ref().unwrap().task_name, "cosmic_tick_health_sample");
        assert!(!result.gpu_anomaly); // facade path is fast
        assert!(result.quantum.quantum_ratio > 0.0);
        assert!(result.quantum.proposal_generated);
        assert!(result.kardashev.is_some());
        assert!(core.gpu_status().dispatch_count > before_g);
        assert!(core.quantum_status().total_weight_updates > before_q);
        assert!(core.kardashev_status().cycle_count > before_k);
        assert!(core.recovery_status().heartbeat_count > before_r);
    }

    #[test]
    fn extended_live_status_snapshot() {
        let core = launch_one_organism_core();
        let s = core.extended_live_status();
        assert!(s.cosmic_loop_ready);
        assert!(!s.active_role.is_empty());
        assert!(s.shared_valence > 0.7);
    }
}
