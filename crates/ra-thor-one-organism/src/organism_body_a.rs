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
    /// Isolation after optional fleet propagation (None if no fleet attached).
    pub fleet_isolation_after: Option<String>,
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
    /// Optional fleet — Medium+ blocks raise progressive isolation.
    pub fleet_surface: Option<UnifiedAgentSurface>,
    pub fleet_agent_id: String,
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
            fleet_surface: None,
            fleet_agent_id: fleet_bind::DEFAULT_ORGANISM_FLEET_AGENT.into(),
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
            // Optional fleet: Medium+ → security signal + progressive isolation
            let fleet_iso = self.propagate_ingest_block_to_fleet(source_label, &scan);

            let report = IngestionAdmissionReport {
                admitted: false, risk_tier: scan.risk_tier.as_str().into(), risk_score: scan.risk_score,
                threats, findings_count: scan.findings.len(), bytes_scanned: scan.bytes_scanned,
                anomaly_reported: true, role_after: self.role_orchestrator.active_role.as_str().into(),
                cosmic_loop_ok: inv.all_hold,
                fleet_isolation_after: fleet_iso,
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
            fleet_isolation_after: None,
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
                cosmic_loop_ok: self.is_cosmic_loop_ready(), fleet_isolation_after: None, message: msg,
            },
            Err(e) => IngestionAdmissionReport {
                admitted: false, risk_tier: "error".into(), risk_score: 0.0, threats: vec![],
                findings_count: 0, bytes_scanned: content.len(), anomaly_reported: false,
                role_after: self.role_orchestrator.active_role.as_str().into(),
                cosmic_loop_ok: self.is_cosmic_loop_ready(), fleet_isolation_after: None, message: format!("{}", e),
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

