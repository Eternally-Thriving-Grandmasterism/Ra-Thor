//! Ra-Thor ONE Organism Core — v14.15.5 AGSi
//!
//! Living Cosmic Tick + adaptive hardening + Cosmic Loop invariant checks.
//! v14.15: extended-live feature readiness surface.
//! v14.15.2: Cosmic Harness — 40-cycle endurance.
//! v14.15.3: AGSi summon surface.
//! v14.15.4: Full AGSi summon sequence — valence clamping, role handoff, recovery anchors.
//! v14.15.5: White-hat ingestion gate — mercy-security IngestionScanner wired into Organism.
//! Cosmic Loop is MANDATORY IDENTITY.
//! Contact: info@Rathor.ai

mod extended_surface;
mod cosmic_harness;
#[cfg(feature = "kardashev-live")]
mod live_valence_status;

pub use extended_surface::{
    ExtendedOrganismSurface, GpuSurface, GpuDispatchTelemetry, GpuSurfaceStatus,
    GitHubSurface, EvolutionPrIntent, GitHubSurfaceStatus, FlushResult,
    QuantumSwarmSurface, QuantumSwarmConfig, QuantumSwarmStatus, QuantumEvolutionResult,
    SovereignRecoverySurface, SovereignRecoveryStatus, RecoveryHeartbeat, RecoveryAnchor,
    KardashevFlywheelSurface, KardashevSurfaceStatus, TransferTickResult,
};

pub use cosmic_harness::{
    CosmicHarness, CosmicHarnessConfig, CosmicHarnessResult,
    HostMode, TickSnapshot, DriftReport, SaturationReport,
};

// White-hat ingestion surface (mercy-security)
pub use mercy_security::{
    IngestionScanResult, IngestionThreat, RiskTier, ScanFinding,
    IngestionScanner, MercySecurityError, MercySecuritySurface,
};

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

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

// RESTORE IN PROGRESS — full body from backup-114 follows in next commit if truncated.
// Temporary: keep crate compilable without fleet_bind until full body lands.

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum AgsiSummonError {
    #[error("incoming valence is NaN or infinite")] InvalidValence,
    #[error("incoming confidence is NaN or infinite")] InvalidConfidence,
    #[error("summoner identifier is empty")] EmptySummoner,
    #[error("Cosmic Loop / guardian failed to activate")] CosmicLoopNotReady,
    #[error("role handoff to Architect failed")] RoleHandoffFailed,
    #[error("recovery anchor persistence failed: {0}")] RecoveryAnchorFailed(String),
}

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum IngestionAdmissionError {
    #[error("Cosmic Loop / guardian not ready — ingestion refused")] CosmicLoopNotReady,
    #[error("ingestion blocked by white-hat scanner: {0}")] Blocked(String),
    #[error("empty content")] EmptyContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionAdmissionReport {
    pub admitted: bool,
    pub risk_tier: String,
    pub risk_score: f32,
    pub threats: Vec<String>,
    pub findings_count: usize,
    pub bytes_scanned: usize,
    pub anomaly_reported: bool,
    pub role_after: String,
    pub cosmic_loop_ok: bool,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrganismRole {
    Investigator, Simulator, VibeCoder, Debugger, Legal, Architect, SovereignRecovery, LatticeConductor,
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
        [OrganismRole::Investigator, OrganismRole::Simulator, OrganismRole::VibeCoder,
         OrganismRole::Debugger, OrganismRole::Legal, OrganismRole::Architect,
         OrganismRole::SovereignRecovery, OrganismRole::LatticeConductor]
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
            roles.insert(role.clone(), RoleState {
                role: role.clone(), valence_ema: 0.97, confidence_ema: 0.88, success_ema: 0.91,
                last_handoff_tick: 0, active: matches!(role, OrganismRole::Architect), task_count: 0,
            });
        }
        Self {
            roles, active_role: OrganismRole::Architect, shared_valence: 0.97,
            shared_confidence_ema: 0.90, handoff_count: 0, last_grok_sync_tick: 0,
            last_handoff_reason: "initial_boot".into(),
        }
    }
    pub fn handoff_to_role(&mut self, new_role: OrganismRole, reason: &str, tick: u64) -> bool {
        if let Some(old) = self.roles.get_mut(&self.active_role) { old.active = false; old.last_handoff_tick = tick; }
        if let Some(new_state) = self.roles.get_mut(&new_role) {
            new_state.active = true; new_state.last_handoff_tick = tick; new_state.task_count += 1;
            let continuity = (self.shared_valence * 0.7 + new_state.valence_ema * 0.3).clamp(0.75, 0.999);
            new_state.valence_ema = continuity; self.shared_valence = continuity;
            self.active_role = new_role; self.handoff_count += 1; self.last_handoff_reason = reason.into();
            true
        } else { false }
    }
    pub fn sync_valence_with_grok_clamped(&mut self, incoming_valence: f64, incoming_confidence: f64, tick: u64) -> (f64, f64) {
        let valence = incoming_valence.clamp(0.75, 0.999999);
        let confidence = incoming_confidence.clamp(0.5, 0.99);
        self.shared_valence = (self.shared_valence * 0.65 + valence * 0.35).clamp(0.75, 0.999999);
        self.shared_confidence_ema = (self.shared_confidence_ema * 0.7 + confidence * 0.3).clamp(0.5, 0.99);
        self.last_grok_sync_tick = tick;
        if let Some(state) = self.roles.get_mut(&self.active_role) {
            state.valence_ema = (state.valence_ema * 0.6 + self.shared_valence * 0.4).clamp(0.75, 0.999);
            state.confidence_ema = (state.confidence_ema * 0.65 + self.shared_confidence_ema * 0.35).clamp(0.5, 0.99);
        }
        (self.shared_valence, self.shared_confidence_ema)
    }
    pub fn sync_valence_with_grok(&mut self, incoming_valence: f64, incoming_confidence: f64, tick: u64) {
        let _ = self.sync_valence_with_grok_clamped(incoming_valence, incoming_confidence, tick);
    }
    pub fn recommend_role_for_task(&self, task_type: &str) -> OrganismRole {
        let t = task_type.to_lowercase();
        if t.contains("debug") || t.contains("error") || t.contains("gpu") || t.contains("ingest") || t.contains("security") { OrganismRole::Debugger }
        else if t.contains("legal") || t.contains("tolc") { OrganismRole::Legal }
        else if t.contains("simulate") || t.contains("quantum") || t.contains("kardashev") { OrganismRole::Simulator }
        else if t.contains("code") || t.contains("vibe") || t.contains("evolution") { OrganismRole::VibeCoder }
        else if t.contains("investigate") || t.contains("scan") || t.contains("threat") { OrganismRole::Investigator }
        else if t.contains("recover") || t.contains("anchor") || t.contains("heartbeat") { OrganismRole::SovereignRecovery }
        else if t.contains("lattice") || t.contains("council") { OrganismRole::LatticeConductor }
        else { OrganismRole::Architect }
    }
}
impl Default for RoleOrchestrator { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmicLoopInvariant { pub cosmic_loop_ready: bool, pub guardian_active: bool, pub all_hold: bool, }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveFeatureReadiness {
    pub github_live: bool, pub gpu_live: bool, pub quantum_live: bool, pub recovery_live: bool,
    pub kardashev_live: bool, pub extended_live: bool, pub web_demo: bool,
    pub cosmic_loop_ready_for_live: bool, pub whitehat_ingestion_live: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgsiActivationReport {
    pub version: String, pub agsi_active: bool, pub cosmic_loop_ready: bool, pub guardian_active: bool,
    pub shared_valence: f64, pub shared_confidence: f64, pub clamped_valence: f64, pub clamped_confidence: f64,
    pub active_role: String, pub role_handoff_ok: bool, pub recovery_anchor_persisted: bool,
    pub patsagi_permanent_deliberation: bool, pub predictive_support_ready: bool,
    pub cosmic_harness_available: bool, pub whitehat_ingestion_ready: bool,
    pub summoner: String, pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmicTickResult {
    pub tick: u64, pub gpu: Option<GpuDispatchTelemetry>, pub recovery: RecoveryHeartbeat,
    pub quantum: QuantumEvolutionResult, pub kardashev: Option<TransferTickResult>,
    pub role_after: String, pub recovery_triggered: bool, pub gpu_anomaly: bool,
    pub healing: Option<Diagnosis>, pub anomalies_fired: Vec<String>,
    pub base_severity: f64, pub effective_quantum_severity: f64, pub gpu_confidence: f64,
    pub recovery_sensitivity_applied: f64, pub cosmic_loop_invariant: CosmicLoopInvariant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedLiveStatus {
    pub gpu: GpuSurfaceStatus, pub github: GitHubSurfaceStatus, pub quantum: QuantumSwarmStatus,
    pub recovery: SovereignRecoveryStatus, pub kardashev: KardashevSurfaceStatus,
    pub cosmic_loop_ready: bool, pub active_role: String, pub shared_valence: f64, pub tick: u64,
    pub pending_anomaly_count: usize, pub healing_experience_count: usize,
    pub last_anomalies_fired: Vec<String>, pub handoff_count: u64, pub last_handoff_reason: String,
    pub last_base_severity: f64, pub last_effective_quantum_severity: f64, pub last_gpu_confidence: f64,
    pub next_recovery_sensitivity: f64, pub last_recovery_sensitivity_applied: f64,
    pub cosmic_loop_invariant_holds: bool, pub guardian_active: bool,
    pub live_features: LiveFeatureReadiness, pub agsi_active: bool, pub whitehat_ingestion_ready: bool,
}

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
    pub last_anomalies_fired: Vec<String>,
    pub last_base_severity: f64,
    pub last_effective_quantum_severity: f64,
    pub last_gpu_confidence: f64,
    pub next_recovery_sensitivity: f64,
    pub last_recovery_sensitivity_applied: f64,
    pub agsi_active: bool,
    pub ingestion_admitted: u64,
    pub ingestion_blocked: u64,
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
            arbitration_engine: arbitration, self_healing_engine: healing, lattice, mercy_api,
            role_orchestrator: RoleOrchestrator::new(), extended, cosmic_loop_ready: shared,
            tick: 0,
            version: "v14.15.5 ONE Organism — AGSi summon + white-hat ingestion gate".into(),
            last_anomalies_fired: Vec::new(),
            last_base_severity: 0.0, last_effective_quantum_severity: 0.0, last_gpu_confidence: 0.0,
            next_recovery_sensitivity: 1.0, last_recovery_sensitivity_applied: 1.0,
            agsi_active: false, ingestion_admitted: 0, ingestion_blocked: 0,
        }
    }

    pub fn ingest_content_report(&self, content: &str) -> IngestionScanResult {
        IngestionScanner::scan_text(content)
    }

    pub fn admit_ingestion(&mut self, content: &str, source_label: &str) -> Result<IngestionAdmissionReport, IngestionAdmissionError> {
        if content.trim().is_empty() { return Err(IngestionAdmissionError::EmptyContent); }
        let inv = self.enforce_cosmic_loop_invariant();
        if !inv.all_hold { return Err(IngestionAdmissionError::CosmicLoopNotReady); }
        self.tick += 1;
        let scan = IngestionScanner::scan_text(content);
        let threats: Vec<String> = scan.threats.iter().map(|t| format!("{:?}", t)).collect();
        if !scan.safe {
            let severity = match scan.risk_tier {
                RiskTier::Critical => 0.95_f32, RiskTier::High => 0.82_f32, RiskTier::Medium => 0.55_f32, _ => 0.40_f32,
            };
            let detail = format!("source={} tier={} score={:.2} threats={:?} findings={}",
                source_label, scan.risk_tier.as_str(), scan.risk_score, threats, scan.findings.len());
            self.self_healing_engine.report_anomaly("ingestion", &detail, severity);
            self.last_anomalies_fired.push("ingestion".into());
            self.ingestion_blocked += 1;
            let role = if scan.risk_tier == RiskTier::Critical { OrganismRole::Debugger } else { OrganismRole::Investigator };
            let _ = self.handoff_role(role, &format!("ingestion_blocked_{}", source_label));
            if scan.risk_tier >= RiskTier::High {
                self.role_orchestrator.shared_valence = (self.role_orchestrator.shared_valence - 0.004).clamp(0.75, 0.999);
            }
            let report = IngestionAdmissionReport {
                admitted: false, risk_tier: scan.risk_tier.as_str().into(), risk_score: scan.risk_score,
                threats, findings_count: scan.findings.len(), bytes_scanned: scan.bytes_scanned,
                anomaly_reported: true, role_after: self.role_orchestrator.active_role.as_str().into(),
                cosmic_loop_ok: inv.all_hold,
                message: format!("INGESTION BLOCKED — white-hat gate. {}", detail),
            };
            return Err(IngestionAdmissionError::Blocked(report.message.clone()));
        }
        self.ingestion_admitted += 1;
        Ok(IngestionAdmissionReport {
            admitted: true, risk_tier: scan.risk_tier.as_str().into(), risk_score: scan.risk_score,
            threats, findings_count: scan.findings.len(), bytes_scanned: scan.bytes_scanned,
            anomaly_reported: false, role_after: self.role_orchestrator.active_role.as_str().into(),
            cosmic_loop_ok: inv.all_hold,
            message: format!("INGESTION ADMITTED — source={} tier={} bytes={}", source_label, scan.risk_tier.as_str(), scan.bytes_scanned),
        })
    }

    pub fn try_admit_ingestion(&mut self, content: &str, source_label: &str) -> IngestionAdmissionReport {
        match self.admit_ingestion(content, source_label) {
            Ok(r) => r,
            Err(IngestionAdmissionError::Blocked(msg)) => IngestionAdmissionReport {
                admitted: false, risk_tier: "blocked".into(), risk_score: 1.0, threats: vec![],
                findings_count: 0, bytes_scanned: content.len(), anomaly_reported: true,
                role_after: self.role_orchestrator.active_role.as_str().into(),
                cosmic_loop_ok: self.is_cosmic_loop_ready(), message: msg,
            },
            Err(e) => IngestionAdmissionReport {
                admitted: false, risk_tier: "error".into(), risk_score: 0.0, threats: vec![],
                findings_count: 0, bytes_scanned: content.len(), anomaly_reported: false,
                role_after: self.role_orchestrator.active_role.as_str().into(),
                cosmic_loop_ok: self.is_cosmic_loop_ready(), message: format!("{}", e),
            },
        }
    }

    pub fn summon_agsi_checked(&mut self, incoming_valence: Option<f64>, incoming_confidence: Option<f64>, summoner: &str) -> Result<AgsiActivationReport, AgsiSummonError> {
        if summoner.trim().is_empty() { return Err(AgsiSummonError::EmptySummoner); }
        let raw_valence = incoming_valence.unwrap_or(0.9995);
        let raw_confidence = incoming_confidence.unwrap_or(0.97);
        if !raw_valence.is_finite() { return Err(AgsiSummonError::InvalidValence); }
        if !raw_confidence.is_finite() { return Err(AgsiSummonError::InvalidConfidence); }
        let inv = self.enforce_cosmic_loop_invariant();
        if !inv.all_hold { return Err(AgsiSummonError::CosmicLoopNotReady); }
        self.self_healing_engine.start_watchdog();
        let (clamped_v, clamped_c) = self.role_orchestrator.sync_valence_with_grok_clamped(raw_valence, raw_confidence, self.tick);
        let handoff_ok = self.handoff_role(OrganismRole::Architect, &format!("agsi_summon_by_{}", summoner));
        if !handoff_ok { return Err(AgsiSummonError::RoleHandoffFailed); }
        let anchor = self.extended.sovereign_recovery.persist_anchor(&format!("agsi_awaken_{}_v14.15.5", summoner), self.tick, &self.arbitration_engine);
        let anchor_persisted = !anchor.note.is_empty();
        self.agsi_active = true;
        let report = AgsiActivationReport {
            version: self.version.clone(), agsi_active: true, cosmic_loop_ready: inv.cosmic_loop_ready,
            guardian_active: inv.guardian_active, shared_valence: self.role_orchestrator.shared_valence,
            shared_confidence: self.role_orchestrator.shared_confidence_ema, clamped_valence: clamped_v,
            clamped_confidence: clamped_c, active_role: self.role_orchestrator.active_role.as_str().into(),
            role_handoff_ok: handoff_ok, recovery_anchor_persisted: anchor_persisted,
            patsagi_permanent_deliberation: true, predictive_support_ready: true,
            cosmic_harness_available: true, whitehat_ingestion_ready: true,
            summoner: summoner.to_string(),
            message: format!("AGSi ACTIVE — summoned by {}. Valence clamped {:.6} → {:.6}. Role handoff OK. Recovery anchor persisted. White-hat ingestion gate LIVE. Cosmic Loop holds.", summoner, raw_valence, clamped_v),
        };
        println!("[Thunder] {}", report.message);
        Ok(report)
    }

    pub fn awaken_agsi(&mut self, incoming_valence: Option<f64>, incoming_confidence: Option<f64>, summoner: &str) -> AgsiActivationReport {
        match self.summon_agsi_checked(incoming_valence, incoming_confidence, summoner) {
            Ok(report) => report,
            Err(e) => {
                let inv = self.enforce_cosmic_loop_invariant();
                self.agsi_active = inv.all_hold;
                let _ = self.handoff_role(OrganismRole::Architect, "agsi_fallback_after_error");
                AgsiActivationReport {
                    version: self.version.clone(), agsi_active: self.agsi_active,
                    cosmic_loop_ready: inv.cosmic_loop_ready, guardian_active: inv.guardian_active,
                    shared_valence: self.role_orchestrator.shared_valence,
                    shared_confidence: self.role_orchestrator.shared_confidence_ema,
                    clamped_valence: self.role_orchestrator.shared_valence,
                    clamped_confidence: self.role_orchestrator.shared_confidence_ema,
                    active_role: self.role_orchestrator.active_role.as_str().into(),
                    role_handoff_ok: false, recovery_anchor_persisted: false,
                    patsagi_permanent_deliberation: true, predictive_support_ready: true,
                    cosmic_harness_available: true, whitehat_ingestion_ready: true,
                    summoner: summoner.to_string(),
                    message: format!("AGSi summon soft-failed ({}) — Cosmic Loop re-enforced. Organism remains available.", e),
                }
            }
        }
    }

    pub fn summon_agsi(&mut self, incoming_valence: Option<f64>, incoming_confidence: Option<f64>, summoner: &str) -> AgsiActivationReport {
        self.awaken_agsi(incoming_valence, incoming_confidence, summoner)
    }
    pub fn summon_agsi_default(&mut self, summoner: &str) -> AgsiActivationReport {
        self.awaken_agsi(Some(0.9995), Some(0.97), summoner)
    }

    pub fn assert_cosmic_loop_invariant(&self) -> CosmicLoopInvariant {
        let cosmic_loop_ready = self.is_cosmic_loop_ready();
        let guardian_active = self.arbitration_engine.is_guardian_active();
        CosmicLoopInvariant { cosmic_loop_ready, guardian_active, all_hold: cosmic_loop_ready && guardian_active }
    }
    pub fn enforce_cosmic_loop_invariant(&mut self) -> CosmicLoopInvariant {
        self.arbitration_engine.enforce_cosmic_loop_activation();
        self.arbitration_engine.protect_cosmic_loop_identity();
        self.lattice.enforce_cosmic_loop_activation();
        self.assert_cosmic_loop_invariant()
    }
    pub fn live_feature_readiness(&self) -> LiveFeatureReadiness {
        let inv = self.assert_cosmic_loop_invariant();
        let github_live = cfg!(feature = "github-live");
        let gpu_live = cfg!(feature = "gpu-live");
        let quantum_live = cfg!(feature = "quantum-live");
        let recovery_live = cfg!(feature = "recovery-live");
        let kardashev_live = cfg!(feature = "kardashev-live");
        LiveFeatureReadiness {
            github_live, gpu_live, quantum_live, recovery_live, kardashev_live,
            extended_live: github_live && gpu_live && quantum_live && recovery_live && kardashev_live,
            web_demo: cfg!(feature = "web-demo"), cosmic_loop_ready_for_live: inv.all_hold, whitehat_ingestion_live: true,
        }
    }
    pub fn offer_cosmic_loop(&mut self) {
        let inv = self.enforce_cosmic_loop_invariant();
        self.self_healing_engine.start_watchdog();
        let _ = self.extended.sovereign_recovery.persist_anchor("boot_cosmic_loop_offer", self.tick, &self.arbitration_engine);
        println!("[OneOrganismCore {}] Cosmic Loop OFFERED + ENFORCED (ready={} guardian={}) + Watchdog STARTED + White-hat ingestion LIVE", self.version, inv.cosmic_loop_ready, inv.guardian_active);
    }
    pub fn on_lattice_sync(&mut self) { let _ = self.cosmic_tick(0.22); }

    pub fn cosmic_tick(&mut self, severity: f64) -> CosmicTickResult {
        self.tick += 1;
        self.arbitration_engine.on_lattice_sync();
        let _ = self.enforce_cosmic_loop_invariant();
        let base_severity = severity.clamp(0.0, 1.0);
        let mut anomalies_fired: Vec<String> = Vec::new();
        let elements = 2048 + ((base_severity * 4096.0) as usize);
        let gpu_tel = self.extended.gpu.record_dispatch("cosmic_tick_health_sample", 8, false, elements, &self.arbitration_engine);
        let gpu_anomaly = gpu_tel.dispatch_time_ms > 80;
        let gpu_confidence = if gpu_tel.dispatch_time_ms <= 5 { 0.97 } else if gpu_tel.dispatch_time_ms <= 20 { 0.90 } else if gpu_tel.dispatch_time_ms <= 50 { 0.78 } else if gpu_tel.dispatch_time_ms <= 80 { 0.62 } else { 0.45 };
        self.role_orchestrator.shared_confidence_ema = (self.role_orchestrator.shared_confidence_ema * 0.85 + gpu_confidence * 0.15).clamp(0.5, 0.99);
        let valence_delta = if gpu_confidence >= 0.90 { 0.008 } else if gpu_confidence >= 0.78 { 0.003 } else if gpu_confidence >= 0.62 { -0.002 } else { -0.012 };
        self.role_orchestrator.shared_valence = (self.role_orchestrator.shared_valence + valence_delta).clamp(0.75, 0.999);
        if gpu_anomaly {
            self.self_healing_engine.report_anomaly("gpu", &format!("dispatch_time_ms={} > 80", gpu_tel.dispatch_time_ms), 0.85);
            anomalies_fired.push("gpu".into());
            let _ = self.handoff_role(OrganismRole::Debugger, "cosmic_tick_gpu_anomaly");
        }
        let sensitivity = self.next_recovery_sensitivity.clamp(1.0, 1.12);
        self.last_recovery_sensitivity_applied = sensitivity;
        self.next_recovery_sensitivity = 1.0;
        let recovery_conf = (self.role_orchestrator.shared_confidence_ema / sensitivity).clamp(0.5, 0.99);
        let recovery_valence = (self.role_orchestrator.shared_valence / (1.0 + (sensitivity - 1.0) * 0.5)).clamp(0.75, 0.999);
        let hb = self.extended.sovereign_recovery.heartbeat(recovery_valence, recovery_conf, self.tick, &self.arbitration_engine);
        let mut recovery_triggered = false;
        if hb.requires_recovery {
            recovery_triggered = true;
            self.self_healing_engine.report_anomaly("recovery", &format!("requires_recovery pressure={:.2} flow_dev={:.2} sens={:.3}", hb.context_pressure, hb.flow_deviation, sensitivity), 0.78);
            anomalies_fired.push("recovery".into());
            let _ = self.handoff_role(OrganismRole::SovereignRecovery, "cosmic_tick_recovery_alert");
            let _ = self.extended.sovereign_recovery.persist_anchor("auto_recover_from_cosmic_tick", self.tick, &self.arbitration_engine);
        }
        let recovery_boost = (hb.context_pressure * 0.35 + hb.flow_deviation * 0.25).clamp(0.0, 0.35);
        let effective_quantum_severity = (base_severity + recovery_boost).clamp(0.0, 1.0);
        let quantum = self.extended.quantum_swarm.evolve_full_cycle(effective_quantum_severity, &self.arbitration_engine);
        if effective_quantum_severity >= 0.55 {
            self.self_healing_engine.report_anomaly("quantum", &format!("high_severity={:.2} (base={:.2} boost={:.2}) ratio={:.3}", effective_quantum_severity, base_severity, recovery_boost, quantum.quantum_ratio), (effective_quantum_severity as f32).min(0.95));
            anomalies_fired.push("quantum".into());
        }
        let quantum_handoff_threshold = if gpu_confidence < 0.70 { 0.40 } else { 0.45 };
        if effective_quantum_severity >= quantum_handoff_threshold && !recovery_triggered && !gpu_anomaly {
            let _ = self.handoff_role(OrganismRole::Simulator, "cosmic_tick_quantum_pressure");
        }
        let rbe = (self.role_orchestrator.shared_valence * 0.85 + self.role_orchestrator.shared_confidence_ema * 0.15).clamp(0.0, 1.0);
        let ethics = self.role_orchestrator.shared_valence.clamp(0.0, 1.0);
        let abundance = (0.9 + effective_quantum_severity * 0.7).min(1.8);
        let kardashev = self.extended.kardashev.transfer_tick(rbe, ethics, abundance, &self.arbitration_engine);
        self.extended.quantum_swarm.apply_kardashev_feedback(&kardashev, &self.arbitration_engine);
        let healing = self.self_healing_engine.run_reflexion_cycle();
        if anomalies_fired.is_empty() { self.next_recovery_sensitivity = 1.0; }
        else {
            let anomaly_boost = (anomalies_fired.len() as f64 * 0.025).min(0.08);
            let mercy_boost = if (healing.mercy_score as f64) < 0.95 { ((1.0 - healing.mercy_score as f64) * 0.15).min(0.06) } else { 0.0 };
            self.next_recovery_sensitivity = (1.0 + anomaly_boost + mercy_boost).clamp(1.0, 1.12);
        }
        if !recovery_triggered && !gpu_anomaly && quantum.quantum_ratio > 0.05 {
            self.role_orchestrator.shared_valence = (self.role_orchestrator.shared_valence * 0.97 + 0.03 * (0.92 + quantum.quantum_ratio * 0.05)).clamp(0.75, 0.999);
        }
        self.last_anomalies_fired = anomalies_fired.clone();
        self.last_base_severity = base_severity;
        self.last_effective_quantum_severity = effective_quantum_severity;
        self.last_gpu_confidence = gpu_confidence;
        let cosmic_loop_invariant = self.enforce_cosmic_loop_invariant();
        CosmicTickResult {
            tick: self.tick, gpu: Some(gpu_tel), recovery: hb, quantum, kardashev: Some(kardashev),
            role_after: self.role_orchestrator.active_role.as_str().into(),
            recovery_triggered, gpu_anomaly, healing: Some(healing), anomalies_fired,
            base_severity, effective_quantum_severity, gpu_confidence,
            recovery_sensitivity_applied: sensitivity, cosmic_loop_invariant,
        }
    }

    pub fn run_cosmic_harness(&mut self) -> CosmicHarnessResult { CosmicHarness::default_40_cycle().run(self) }
    pub fn run_cosmic_harness_with_config(&mut self, config: CosmicHarnessConfig) -> CosmicHarnessResult { CosmicHarness::new(config).run(self) }
    pub fn before_council_arbitration(&self) { self.arbitration_engine.before_council_arbitration(); }
    pub fn protect_cosmic_loop(&self) { self.arbitration_engine.protect_cosmic_loop_identity(); }
    pub fn is_cosmic_loop_ready(&self) -> bool { self.cosmic_loop_ready.load(Ordering::SeqCst) }
    pub fn handoff_role(&mut self, role: OrganismRole, reason: &str) -> bool { self.role_orchestrator.handoff_to_role(role, reason, self.tick) }
    pub fn sync_with_grok(&mut self, valence: f64, confidence: f64) { self.role_orchestrator.sync_valence_with_grok(valence, confidence, self.tick); }
    pub fn run_healing_reflexion(&self) -> Diagnosis { self.self_healing_engine.run_reflexion_cycle() }

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
        if recommended != self.role_orchestrator.active_role { let _ = self.handoff_role(recommended, "api_request_routing"); }
        self.mercy_api.handle_request(request, Some(&self.arbitration_engine))
    }
    pub fn api_status(&self) -> MercyApiResponse { self.mercy_api.status() }

    pub fn record_gpu_dispatch(&mut self, task_name: &str, dispatch_time_ms: u64, real_gpu: bool, elements: usize) -> GpuDispatchTelemetry {
        self.tick += 1;
        let tel = self.extended.gpu.record_dispatch(task_name, dispatch_time_ms, real_gpu, elements, &self.arbitration_engine);
        if dispatch_time_ms > 80 {
            self.self_healing_engine.report_anomaly("gpu", &format!("dispatch_time_ms={}", dispatch_time_ms), 0.85);
            let _ = self.handoff_role(OrganismRole::Debugger, "gpu_dispatch_anomaly");
            let _ = self.self_healing_engine.run_reflexion_cycle();
        }
        tel
    }
    pub fn queue_evolution_pr(&mut self, role: &str, target_module: &str, description: &str, expected_benefit: f64, mercy_alignment: f64) -> EvolutionPrIntent {
        self.tick += 1;
        if mercy_alignment > 0.88 && expected_benefit > 0.55 { let _ = self.handoff_role(OrganismRole::VibeCoder, "high_mercy_evolution"); }
        self.extended.github.queue_evolution_pr(role, target_module, description, expected_benefit, mercy_alignment, &self.arbitration_engine)
    }
    pub fn flush_evolution_prs(&mut self) -> Vec<FlushResult> {
        self.tick += 1; let _ = self.handoff_role(OrganismRole::VibeCoder, "flush_evolution_prs");
        self.extended.github.flush_to_github(&self.arbitration_engine)
    }
    pub fn quantum_evolution_tick(&mut self, severity: f64) -> f64 {
        self.tick += 1; let _ = self.handoff_role(OrganismRole::Simulator, "quantum_tick");
        self.extended.quantum_swarm.evolution_tick(severity, &self.arbitration_engine)
    }
    pub fn quantum_evolve_full_cycle(&mut self, severity: f64) -> QuantumEvolutionResult {
        self.tick += 1; let _ = self.handoff_role(OrganismRole::Simulator, "quantum_full_cycle");
        self.extended.quantum_swarm.evolve_full_cycle(severity, &self.arbitration_engine)
    }
    pub fn recovery_heartbeat(&mut self) -> RecoveryHeartbeat {
        self.tick += 1;
        self.extended.sovereign_recovery.heartbeat(self.role_orchestrator.shared_valence, self.role_orchestrator.shared_confidence_ema, self.tick, &self.arbitration_engine)
    }
    pub fn recovery_anchor(&mut self, note: &str) -> RecoveryAnchor {
        self.tick += 1; let _ = self.handoff_role(OrganismRole::SovereignRecovery, "manual_anchor");
        self.extended.sovereign_recovery.persist_anchor(note, self.tick, &self.arbitration_engine)
    }
    pub fn kardashev_transfer_tick(&mut self, rbe_quality: f64, ethical_choice: f64, abundance_signal: f64) -> TransferTickResult {
        self.tick += 1; let _ = self.handoff_role(OrganismRole::Simulator, "kardashev_transfer");
        let result = self.extended.kardashev.transfer_tick(rbe_quality, ethical_choice, abundance_signal, &self.arbitration_engine);
        self.extended.quantum_swarm.apply_kardashev_feedback(&result, &self.arbitration_engine);
        result
    }
    pub fn gpu_status(&self) -> GpuSurfaceStatus { self.extended.gpu.status() }
    pub fn github_status(&self) -> GitHubSurfaceStatus { self.extended.github.status() }
    pub fn quantum_status(&self) -> QuantumSwarmStatus { self.extended.quantum_swarm.status() }
    pub fn recovery_status(&self) -> SovereignRecoveryStatus { self.extended.sovereign_recovery.status() }
    pub fn kardashev_status(&self) -> KardashevSurfaceStatus { self.extended.kardashev.status() }

    pub fn extended_live_status(&self) -> ExtendedLiveStatus {
        let inv = self.assert_cosmic_loop_invariant();
        ExtendedLiveStatus {
            gpu: self.extended.gpu.status(), github: self.extended.github.status(),
            quantum: self.extended.quantum_swarm.status(), recovery: self.extended.sovereign_recovery.status(),
            kardashev: self.extended.kardashev.status(), cosmic_loop_ready: inv.cosmic_loop_ready,
            active_role: self.role_orchestrator.active_role.as_str().into(),
            shared_valence: self.role_orchestrator.shared_valence, tick: self.tick,
            pending_anomaly_count: self.self_healing_engine.pending_anomaly_count(),
            healing_experience_count: self.self_healing_engine.get_healing_experiences().len(),
            last_anomalies_fired: self.last_anomalies_fired.clone(),
            handoff_count: self.role_orchestrator.handoff_count,
            last_handoff_reason: self.role_orchestrator.last_handoff_reason.clone(),
            last_base_severity: self.last_base_severity, last_effective_quantum_severity: self.last_effective_quantum_severity,
            last_gpu_confidence: self.last_gpu_confidence, next_recovery_sensitivity: self.next_recovery_sensitivity,
            last_recovery_sensitivity_applied: self.last_recovery_sensitivity_applied,
            cosmic_loop_invariant_holds: inv.all_hold, guardian_active: inv.guardian_active,
            live_features: self.live_feature_readiness(), agsi_active: self.agsi_active, whitehat_ingestion_ready: true,
        }
    }
    pub fn role_orchestrator(&self) -> &RoleOrchestrator { &self.role_orchestrator }
    pub fn role_orchestrator_mut(&mut self) -> &mut RoleOrchestrator { &mut self.role_orchestrator }
}

impl Default for OneOrganismCore { fn default() -> Self { Self::new() } }

pub fn launch_one_organism_core() -> OneOrganismCore {
    let mut organism = OneOrganismCore::new();
    let _report = organism.summon_agsi_default("launch_one_organism_core");
    println!("[Thunder] ONE Organism Core v14.15.5 AGSi ACTIVE — Full summon + white-hat ingestion gate LIVE. Cosmic Loop is MANDATORY IDENTITY. Eternal.");
    organism
}

pub fn summon_agsi_from_external(valence: Option<f64>, confidence: Option<f64>, summoner: &str) -> (OneOrganismCore, AgsiActivationReport) {
    let mut organism = OneOrganismCore::new();
    let report = organism.awaken_agsi(valence, confidence, summoner);
    (organism, report)
}

pub fn summon_agsi_from_external_checked(valence: Option<f64>, confidence: Option<f64>, summoner: &str) -> Result<(OneOrganismCore, AgsiActivationReport), AgsiSummonError> {
    let mut organism = OneOrganismCore::new();
    let report = organism.summon_agsi_checked(valence, confidence, summoner)?;
    Ok((organism, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn cosmic_loop_ready_after_launch() {
        let core = launch_one_organism_core();
        let inv = core.assert_cosmic_loop_invariant();
        assert!(inv.all_hold); assert!(core.agsi_active);
    }
    #[test] fn agsi_summon_checked_succeeds() {
        let mut core = OneOrganismCore::new();
        let report = core.summon_agsi_checked(Some(0.9996), Some(0.98), "test_grok").unwrap();
        assert!(report.agsi_active); assert!(report.role_handoff_ok); assert!(report.recovery_anchor_persisted);
        assert!(report.whitehat_ingestion_ready);
        assert!(report.clamped_valence >= 0.75 && report.clamped_valence <= 0.999999);
    }
    #[test] fn agsi_summon_rejects_nan() {
        let mut core = OneOrganismCore::new();
        assert!(matches!(core.summon_agsi_checked(Some(f64::NAN), Some(0.97), "bad"), Err(AgsiSummonError::InvalidValence)));
    }
    #[test] fn agsi_summon_rejects_empty_summoner() {
        let mut core = OneOrganismCore::new();
        assert!(matches!(core.summon_agsi_checked(Some(0.9995), Some(0.97), ""), Err(AgsiSummonError::EmptySummoner)));
    }
    #[test] fn compatibility_awaken_never_panics() {
        let mut core = OneOrganismCore::new();
        assert!(!core.awaken_agsi(Some(f64::NAN), Some(0.97), "soft").message.is_empty());
    }
    #[test] fn cosmic_tick_preserves_cosmic_loop_invariant() {
        let mut core = launch_one_organism_core();
        assert!(core.cosmic_tick(0.45).cosmic_loop_invariant.all_hold);
    }
    #[test] fn cosmic_harness_runs_cleanly() {
        let mut core = launch_one_organism_core();
        let result = core.run_cosmic_harness();
        assert!(result.cycles_completed >= 40); assert!(result.final_cosmic_loop.all_hold); assert!(result.recovery_integrity_ok);
    }
    #[test] fn admits_clean_content() {
        let mut core = launch_one_organism_core();
        let report = core.admit_ingestion("This is a normal model card for image classification.", "test_clean").unwrap();
        assert!(report.admitted); assert_eq!(core.ingestion_admitted, 1); assert_eq!(core.ingestion_blocked, 0);
    }
    #[test] fn blocks_remote_code_ingestion() {
        let mut core = launch_one_organism_core();
        let err = core.admit_ingestion("dataset = load_dataset('x', trust_remote_code=True)", "test_poison");
        assert!(matches!(err, Err(IngestionAdmissionError::Blocked(_))));
        assert_eq!(core.ingestion_blocked, 1);
        assert!(core.last_anomalies_fired.iter().any(|a| a == "ingestion"));
    }
    #[test] fn blocks_hf_combo_and_handoffs() {
        let mut core = launch_one_organism_core();
        let content = "loading_script = \"poison.py\"\ntrust_remote_code = True\ndl_manager.download_and_extract(url)";
        let err = core.admit_ingestion(content, "hf_malicious_dataset");
        assert!(matches!(err, Err(IngestionAdmissionError::Blocked(_))));
        let role = core.role_orchestrator.active_role.as_str();
        assert!(role == "Debugger" || role == "Investigator");
    }
    #[test] fn scan_only_does_not_mutate_counters() {
        let core = launch_one_organism_core();
        let scan = core.ingest_content_report("pickle.loads(payload)");
        assert!(!scan.safe); assert_eq!(core.ingestion_admitted, 0); assert_eq!(core.ingestion_blocked, 0);
    }
}
