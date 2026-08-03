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
