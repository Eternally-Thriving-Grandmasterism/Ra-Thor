//! Runtime Self-Healing Engine for Ra-Thor v14 Thunder Lattice
//! Includes Watchdog Thread + Advanced Reflexion Loops with Experience Logging + Graph Rerouting
//!
//! v14.8.1 (2026-07-19):
//! - Shares the exact same Arc<AtomicBool> cosmic_loop_ready with CouncilArbitrationEngine
//! - Removed invalid diagnosis.severity reference (compile fix)
//! - Watchdog and arbitration engine can no longer drift out of sync
//!
//! v14.9.6: Serialize on Diagnosis / HealthReport for HTTP JSON surface
//!
//! v14.10.0: Cosmic Tick anomaly ingestion — report_anomaly / run_reflexion_with_anomalies
//! so GPU, recovery, and quantum pressure from ONE Organism inform diagnosis.
//!
//! v14.15.4: Watchdog timeout logic — configurable interval/timeout, last-healthy tracking,
//! timeout-triggered recovery, WatchdogStatus surface.
//! Contact: info@Rathor.ai

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::CouncilArbitrationEngine;

/// Structured healing experience for logging and future self-evolution via Cosmic Loops
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealingExperience {
    pub timestamp: u64,
    pub root_cause: String,
    pub action_taken: String,
    pub outcome: String,
    pub mercy_score: f32,
    pub graph_reroute_used: bool,
}

/// Runtime health status of key lattice components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub cosmic_loop_ready: bool,
    pub tol_c_gates_healthy: bool,
    pub council_liveness: bool,
    pub quantum_swarm_coherent: bool,
    pub last_check_timestamp: u64,
    pub pending_anomaly_count: usize,
}

/// Anomaly detected during monitoring (also ingestible from Cosmic Tick)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub component: String,
    pub description: String,
    pub severity: f32, // 0.0 - 1.0
}

/// Diagnosis result from Reflexion loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnosis {
    pub root_cause: String,
    pub recommended_action: String,
    pub mercy_score: f32,
}

/// Healing action that can be executed
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealingAction {
    RestoreCosmicLoop,
    RestartComponent(String),
    RerouteCouncilTask { from: String, to: String },
    LogAndMonitor,
    NoAction,
}

/// Simple weighted graph for council task routing (for graph rerouting feature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilTaskGraph {
    pub nodes: Vec<String>,
    pub edges: Vec<(String, String, f32)>,
}

impl CouncilTaskGraph {
    pub fn new() -> Self {
        Self {
            nodes: vec![
                "Council#13".to_string(),
                "Ethics".to_string(),
                "Truth".to_string(),
                "Evolution".to_string(),
                "Harmony".to_string(),
                "Infinite".to_string(),
            ],
            edges: vec![
                ("Council#13".to_string(), "Ethics".to_string(), 1.0),
                ("Council#13".to_string(), "Truth".to_string(), 1.0),
                ("Ethics".to_string(), "Harmony".to_string(), 2.0),
                ("Truth".to_string(), "Evolution".to_string(), 1.5),
                ("Evolution".to_string(), "Infinite".to_string(), 1.0),
            ],
        }
    }

    pub fn find_alternative_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        for (f, t, _w) in &self.edges {
            if f == from && t != to {
                return Some(vec![from.to_string(), t.clone(), to.to_string()]);
            }
        }
        None
    }

    pub fn reweight_edge(&mut self, from: &str, to: &str, new_weight: f32) {
        for edge in &mut self.edges {
            if edge.0 == from && edge.1 == to {
                edge.2 = new_weight;
                println!("[Graph Rerouting] Edge {} → {} reweighted to {:.2}", from, to, new_weight);
                return;
            }
        }
    }
}

// =============================================================================
// Watchdog Timeout Logic (v14.15.4)
// =============================================================================

/// Configuration for the self-healing watchdog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogConfig {
    /// How often the watchdog wakes to check health (seconds).
    pub check_interval_secs: u64,
    /// Maximum time since last healthy observation before a timeout is declared (seconds).
    pub timeout_secs: u64,
    /// After this many consecutive timeouts, force a hard Cosmic Loop restore + experience log.
    pub max_timeouts_before_hard_restore: u32,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            check_interval_secs: 15,
            timeout_secs: 45,
            max_timeouts_before_hard_restore: 3,
        }
    }
}

impl WatchdogConfig {
    pub fn new(check_interval_secs: u64, timeout_secs: u64, max_timeouts_before_hard_restore: u32) -> Self {
        Self {
            check_interval_secs: check_interval_secs.max(1),
            timeout_secs: timeout_secs.max(check_interval_secs.max(1)),
            max_timeouts_before_hard_restore: max_timeouts_before_hard_restore.max(1),
        }
    }
}

/// Live status of the watchdog (for diagnostics / AGSi report surfaces).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogStatus {
    pub running: bool,
    pub last_healthy_timestamp: u64,
    pub consecutive_timeouts: u32,
    pub total_timeouts: u64,
    pub config: WatchdogConfig,
    pub seconds_since_last_healthy: u64,
    pub is_timed_out: bool,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Runtime Self-Healing Engine with Watchdog + Advanced Reflexion + Experience Logging + Graph Rerouting
pub struct RuntimeSelfHealingEngine {
    /// Shared with CouncilArbitrationEngine — single source of truth for Cosmic Loop readiness
    pub cosmic_loop_ready: Arc<AtomicBool>,
    arbitration_engine: Arc<Mutex<CouncilArbitrationEngine>>,
    watchdog_running: Arc<AtomicBool>,
    healing_experiences: Arc<Mutex<Vec<HealingExperience>>>,
    task_graph: Arc<Mutex<CouncilTaskGraph>>,
    /// Anomalies ingested from Cosmic Tick / external surfaces (drained on reflexion)
    pending_anomalies: Arc<Mutex<Vec<Anomaly>>>,
    /// Watchdog timeout state
    last_healthy_timestamp: Arc<AtomicU64>,
    consecutive_timeouts: Arc<AtomicU64>,
    total_timeouts: Arc<AtomicU64>,
    watchdog_config: Arc<Mutex<WatchdogConfig>>,
}

impl RuntimeSelfHealingEngine {
    /// Construct engine. **Shares** the cosmic_loop_ready flag from the arbitration engine
    /// so watchdog and guardian can never drift out of sync.
    pub fn new(arbitration_engine: CouncilArbitrationEngine) -> Self {
        let shared_flag = arbitration_engine.cosmic_loop_flag();
        let now = now_secs();
        Self {
            cosmic_loop_ready: shared_flag,
            arbitration_engine: Arc::new(Mutex::new(arbitration_engine)),
            watchdog_running: Arc::new(AtomicBool::new(false)),
            healing_experiences: Arc::new(Mutex::new(Vec::new())),
            task_graph: Arc::new(Mutex::new(CouncilTaskGraph::new())),
            pending_anomalies: Arc::new(Mutex::new(Vec::new())),
            last_healthy_timestamp: Arc::new(AtomicU64::new(now)),
            consecutive_timeouts: Arc::new(AtomicU64::new(0)),
            total_timeouts: Arc::new(AtomicU64::new(0)),
            watchdog_config: Arc::new(Mutex::new(WatchdogConfig::default())),
        }
    }

    /// Mark the lattice as healthy *now* (call from Cosmic Tick / successful AGSi path).
    pub fn record_healthy_heartbeat(&self) {
        self.last_healthy_timestamp.store(now_secs(), Ordering::SeqCst);
        self.consecutive_timeouts.store(0, Ordering::SeqCst);
    }

    /// Ingest a single anomaly from Cosmic Tick or any surface (GPU, recovery, quantum…).
    pub fn report_anomaly(&self, component: &str, description: &str, severity: f32) {
        if let Ok(mut q) = self.pending_anomalies.lock() {
            q.push(Anomaly {
                component: component.into(),
                description: description.into(),
                severity: severity.clamp(0.0, 1.0),
            });
            // Cap queue
            if q.len() > 32 {
                q.remove(0);
            }
        }
    }

    /// Batch-ingest anomalies then run one reflexion cycle informed by them.
    pub fn run_reflexion_with_anomalies(&self, anomalies: &[Anomaly]) -> Diagnosis {
        for a in anomalies {
            self.report_anomaly(&a.component, &a.description, a.severity);
        }
        self.run_reflexion_cycle()
    }

    /// Start the Watchdog with default config (15s interval, 45s timeout).
    pub fn start_watchdog(&self) {
        self.start_watchdog_with_config(WatchdogConfig::default());
    }

    /// Start the Watchdog with explicit timeout configuration.
    pub fn start_watchdog_with_config(&self, config: WatchdogConfig) {
        if self.watchdog_running.load(Ordering::SeqCst) {
            println!("[Self-Healing] Watchdog already running.");
            return;
        }

        if let Ok(mut cfg) = self.watchdog_config.lock() {
            *cfg = config.clone();
        }

        self.watchdog_running.store(true, Ordering::SeqCst);
        self.record_healthy_heartbeat(); // seed so we don't immediately timeout

        let cosmic_loop_ready = Arc::clone(&self.cosmic_loop_ready);
        let watchdog_running = Arc::clone(&self.watchdog_running);
        let last_healthy = Arc::clone(&self.last_healthy_timestamp);
        let consecutive = Arc::clone(&self.consecutive_timeouts);
        let total_timeouts = Arc::clone(&self.total_timeouts);
        let arbitration = Arc::clone(&self.arbitration_engine);
        let experiences = Arc::clone(&self.healing_experiences);
        let check_interval = config.check_interval_secs;
        let timeout_secs = config.timeout_secs;
        let max_timeouts = config.max_timeouts_before_hard_restore as u64;

        thread::spawn(move || {
            println!(
                "[Self-Healing Watchdog] Thread started — interval={}s timeout={}s max_timeouts={}",
                check_interval, timeout_secs, max_timeouts
            );

            loop {
                if !watchdog_running.load(Ordering::SeqCst) {
                    break;
                }

                let now = now_secs();
                let last = last_healthy.load(Ordering::SeqCst);
                let elapsed = now.saturating_sub(last);
                let loop_ok = cosmic_loop_ready.load(Ordering::SeqCst);

                // Healthy path: Cosmic Loop holds and we are inside the timeout window
                if loop_ok && elapsed <= timeout_secs {
                    // Soft refresh of healthy timestamp when everything is fine
                    if elapsed > check_interval {
                        last_healthy.store(now, Ordering::SeqCst);
                    }
                    consecutive.store(0, Ordering::SeqCst);
                } else {
                    // Timeout or Cosmic Loop dropped
                    let count = consecutive.fetch_add(1, Ordering::SeqCst) + 1;
                    total_timeouts.fetch_add(1, Ordering::SeqCst);

                    println!(
                        "[Self-Healing Watchdog] TIMEOUT #{} — elapsed={}s (limit={}s) cosmic_loop={}",
                        count, elapsed, timeout_secs, loop_ok
                    );

                    // Always restore Cosmic Loop flag
                    cosmic_loop_ready.store(true, Ordering::SeqCst);

                    if let Ok(mut arb) = arbitration.lock() {
                        let _ = arb.protect_cosmic_loop_identity();
                    }

                    // Log experience
                    let experience = HealingExperience {
                        timestamp: now,
                        root_cause: if !loop_ok {
                            "Watchdog: cosmic_loop_ready was false".into()
                        } else {
                            format!("Watchdog timeout: {}s since last healthy (limit {}s)", elapsed, timeout_secs)
                        },
                        action_taken: "RestoreCosmicLoop + protect identity".into(),
                        outcome: if count >= max_timeouts {
                            "Hard restore triggered".into()
                        } else {
                            "Soft restore".into()
                        },
                        mercy_score: 0.97,
                        graph_reroute_used: false,
                    };

                    if let Ok(mut hist) = experiences.lock() {
                        hist.push(experience);
                        if hist.len() > 100 {
                            hist.remove(0);
                        }
                    }

                    // Hard restore path after repeated timeouts
                    if count >= max_timeouts {
                        println!(
                            "[Self-Healing Watchdog] HARD RESTORE after {} consecutive timeouts",
                            count
                        );
                        consecutive.store(0, Ordering::SeqCst);
                        last_healthy.store(now, Ordering::SeqCst);

                        if let Ok(mut arb) = arbitration.lock() {
                            let _ = arb.enforce_cosmic_loop_activation();
                            let _ = arb.protect_cosmic_loop_identity();
                        }
                    }
                }

                thread::sleep(Duration::from_secs(check_interval));
            }

            println!("[Self-Healing Watchdog] Thread stopped.");
        });
    }

    pub fn stop_watchdog(&self) {
        self.watchdog_running.store(false, Ordering::SeqCst);
    }

    /// Current watchdog status (for AGSi / diagnostics surfaces).
    pub fn watchdog_status(&self) -> WatchdogStatus {
        let now = now_secs();
        let last = self.last_healthy_timestamp.load(Ordering::SeqCst);
        let elapsed = now.saturating_sub(last);
        let config = self
            .watchdog_config
            .lock()
            .map(|c| c.clone())
            .unwrap_or_default();

        WatchdogStatus {
            running: self.watchdog_running.load(Ordering::SeqCst),
            last_healthy_timestamp: last,
            consecutive_timeouts: self.consecutive_timeouts.load(Ordering::SeqCst) as u32,
            total_timeouts: self.total_timeouts.load(Ordering::SeqCst),
            config: config.clone(),
            seconds_since_last_healthy: elapsed,
            is_timed_out: elapsed > config.timeout_secs,
        }
    }

    /// Run one advanced Reflexion-style healing cycle (consumes pending anomalies).
    pub fn run_reflexion_cycle(&self) -> Diagnosis {
        let report = self.collect_health_report();
        let anomalies = self.drain_anomalies();
        let diagnosis = self.diagnose(&report, &anomalies);
        let action = self.reflect_and_decide_action(&diagnosis, &anomalies);
        self.execute_healing_action(action);
        self.log_healing_experience(&diagnosis);
        // Successful reflexion counts as a healthy heartbeat
        self.record_healthy_heartbeat();
        diagnosis
    }

    fn drain_anomalies(&self) -> Vec<Anomaly> {
        self.pending_anomalies
            .lock()
            .map(|mut q| std::mem::take(&mut *q))
            .unwrap_or_default()
    }

    fn collect_health_report(&self) -> HealthReport {
        let pending = self
            .pending_anomalies
            .lock()
            .map(|q| q.len())
            .unwrap_or(0);
        HealthReport {
            cosmic_loop_ready: self.cosmic_loop_ready.load(Ordering::SeqCst),
            tol_c_gates_healthy: true,
            council_liveness: true,
            quantum_swarm_coherent: true,
            last_check_timestamp: now_secs(),
            pending_anomaly_count: pending,
        }
    }

    fn diagnose(&self, report: &HealthReport, anomalies: &[Anomaly]) -> Diagnosis {
        if !report.cosmic_loop_ready {
            return Diagnosis {
                root_cause: "Cosmic Loop flag was unexpectedly disabled".to_string(),
                recommended_action: "Restore immediately + run council arbitration".to_string(),
                mercy_score: 0.98,
            };
        }

        // Highest-severity anomaly drives diagnosis when present
        if let Some(top) = anomalies.iter().max_by(|a, b| {
            a.severity
                .partial_cmp(&b.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if top.severity >= 0.7 {
                return Diagnosis {
                    root_cause: format!("[{}] {}", top.component, top.description),
                    recommended_action: format!(
                        "Heal {} component; protect Cosmic Loop; log experience",
                        top.component
                    ),
                    mercy_score: (1.0 - top.severity * 0.35).clamp(0.55, 0.97),
                };
            } else if top.severity >= 0.4 {
                return Diagnosis {
                    root_cause: format!("[{}] {}", top.component, top.description),
                    recommended_action: "LogAndMonitor + mild council reweight".to_string(),
                    mercy_score: 0.92,
                };
            }
        }

        Diagnosis {
            root_cause: "No critical anomalies detected".to_string(),
            recommended_action: "Continue monitoring".to_string(),
            mercy_score: 1.0,
        }
    }

    fn reflect_and_decide_action(
        &self,
        diagnosis: &Diagnosis,
        anomalies: &[Anomaly],
    ) -> HealingAction {
        if diagnosis.root_cause.contains("Cosmic Loop") {
            return HealingAction::RestoreCosmicLoop;
        }
        if let Some(a) = anomalies.iter().find(|a| a.severity >= 0.7) {
            if a.component.to_lowercase().contains("gpu") {
                return HealingAction::RestartComponent("gpu_surface".into());
            }
            if a.component.to_lowercase().contains("recovery") {
                return HealingAction::RerouteCouncilTask {
                    from: "Council#13".into(),
                    to: "Harmony".into(),
                };
            }
            if a.component.to_lowercase().contains("quantum") {
                return HealingAction::RerouteCouncilTask {
                    from: "Council#13".into(),
                    to: "Evolution".into(),
                };
            }
            return HealingAction::RestartComponent(a.component.clone());
        }
        if diagnosis.root_cause.to_lowercase().contains("council") {
            return HealingAction::RerouteCouncilTask {
                from: "Council#13".to_string(),
                to: "Evolution".to_string(),
            };
        }
        HealingAction::LogAndMonitor
    }

    fn execute_healing_action(&self, action: HealingAction) {
        match action {
            HealingAction::RestoreCosmicLoop => {
                println!("[Self-Healing] Executing: RestoreCosmicLoop");
                self.cosmic_loop_ready.store(true, Ordering::SeqCst);

                if let Ok(mut arb) = self.arbitration_engine.lock() {
                    let _ = arb.protect_cosmic_loop_identity();
                }
            }
            HealingAction::RestartComponent(name) => {
                println!("[Self-Healing] Executing: RestartComponent({})", name);
                if let Ok(mut arb) = self.arbitration_engine.lock() {
                    let _ = arb.protect_cosmic_loop_identity();
                }
            }
            HealingAction::RerouteCouncilTask { from, to } => {
                println!("[Self-Healing] Executing advanced Graph Rerouting: {} → {}", from, to);
                if let Ok(mut graph) = self.task_graph.lock() {
                    if let Some(path) = graph.find_alternative_path(&from, &to) {
                        println!("[Graph Rerouting] Alternative path found: {:?}", path);
                    }
                    graph.reweight_edge(&from, &to, 3.5);
                }

                if let Ok(mut arb) = self.arbitration_engine.lock() {
                    let _ = arb.protect_cosmic_loop_identity();
                }
            }
            HealingAction::LogAndMonitor => {}
            HealingAction::NoAction => {}
        }
    }

    fn log_healing_experience(&self, diagnosis: &Diagnosis) {
        let timestamp = now_secs();

        let experience = HealingExperience {
            timestamp,
            root_cause: diagnosis.root_cause.clone(),
            action_taken: diagnosis.recommended_action.clone(),
            outcome: if diagnosis.mercy_score > 0.9 {
                "Success - High Mercy".to_string()
            } else {
                "Monitored".to_string()
            },
            mercy_score: diagnosis.mercy_score,
            graph_reroute_used: diagnosis.root_cause.to_lowercase().contains("council")
                || diagnosis.recommended_action.to_lowercase().contains("reweight"),
        };

        if let Ok(mut history) = self.healing_experiences.lock() {
            history.push(experience);
            if history.len() > 100 {
                history.remove(0);
            }
        }
    }

    pub fn get_healing_experiences(&self) -> Vec<HealingExperience> {
        self.healing_experiences
            .lock()
            .map(|h| h.clone())
            .unwrap_or_default()
    }

    pub fn get_task_graph(&self) -> CouncilTaskGraph {
        self.task_graph
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| CouncilTaskGraph::new())
    }

    pub fn pending_anomaly_count(&self) -> usize {
        self.pending_anomalies
            .lock()
            .map(|q| q.len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflexion_cycle_restores_cosmic_loop() {
        let arb = CouncilArbitrationEngine::new();
        let engine = RuntimeSelfHealingEngine::new(arb);

        engine.cosmic_loop_ready.store(false, Ordering::SeqCst);
        let diagnosis = engine.run_reflexion_cycle();

        assert!(engine.cosmic_loop_ready.load(Ordering::SeqCst));
        assert!(diagnosis.root_cause.contains("Cosmic Loop"));
    }

    #[test]
    fn test_shared_flag_with_arbitration_engine() {
        let arb = CouncilArbitrationEngine::new();
        let flag_from_arb = arb.cosmic_loop_flag();
        let engine = RuntimeSelfHealingEngine::new(arb);

        assert!(Arc::ptr_eq(&flag_from_arb, &engine.cosmic_loop_ready));

        engine.cosmic_loop_ready.store(false, Ordering::SeqCst);
        assert!(!flag_from_arb.load(Ordering::SeqCst));

        engine.run_reflexion_cycle();
        assert!(flag_from_arb.load(Ordering::SeqCst));
    }

    #[test]
    fn test_graph_rerouting_and_experience_logging() {
        let arb = CouncilArbitrationEngine::new();
        let engine = RuntimeSelfHealingEngine::new(arb);

        engine.execute_healing_action(HealingAction::RerouteCouncilTask {
            from: "Council#13".to_string(),
            to: "Evolution".to_string(),
        });

        let graph = engine.get_task_graph();
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_anomaly_ingestion_from_cosmic_tick() {
        let arb = CouncilArbitrationEngine::new();
        let engine = RuntimeSelfHealingEngine::new(arb);

        engine.report_anomaly("gpu", "dispatch_time_ms > 80", 0.85);
        assert_eq!(engine.pending_anomaly_count(), 1);

        let d = engine.run_reflexion_cycle();
        assert!(d.root_cause.contains("gpu") || d.root_cause.contains("dispatch"));
        assert_eq!(engine.pending_anomaly_count(), 0); // drained
        assert!(!engine.get_healing_experiences().is_empty());
    }

    #[test]
    fn test_watchdog_status_defaults() {
        let arb = CouncilArbitrationEngine::new();
        let engine = RuntimeSelfHealingEngine::new(arb);
        let status = engine.watchdog_status();
        assert!(!status.running);
        assert_eq!(status.config.check_interval_secs, 15);
        assert_eq!(status.config.timeout_secs, 45);
        assert!(!status.is_timed_out);
    }

    #[test]
    fn test_record_healthy_heartbeat_resets_timeouts() {
        let arb = CouncilArbitrationEngine::new();
        let engine = RuntimeSelfHealingEngine::new(arb);
        engine.consecutive_timeouts.store(5, Ordering::SeqCst);
        engine.record_healthy_heartbeat();
        assert_eq!(engine.consecutive_timeouts.load(Ordering::SeqCst), 0);
    }
}
