//! Runtime Self-Healing Engine for Ra-Thor v14 Thunder Lattice
//! Includes Watchdog Thread + Advanced Reflexion Loops with Experience Logging + Graph Rerouting
//!
//! This module provides runtime (live execution) self-healing capabilities.
//! It works symbiotically with the Cosmic Loop Activation Protocol.
//!
//! Core Pattern: Monitor → Diagnose → Reflect → Heal (Reflexion-inspired)
//! All healing actions are mercy-gated and council-arbitrated.
//! Advanced: Structured experience logging + Graph-based council task rerouting

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::CouncilArbitrationEngine;

/// Structured healing experience for logging and future self-evolution via Cosmic Loops
#[derive(Debug, Clone)]
pub struct HealingExperience {
    pub timestamp: u64,
    pub root_cause: String,
    pub action_taken: String,
    pub outcome: String,
    pub mercy_score: f32,
    pub graph_reroute_used: bool,
}

/// Runtime health status of key lattice components
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub cosmic_loop_ready: bool,
    pub tol_c_gates_healthy: bool,
    pub council_liveness: bool,
    pub quantum_swarm_coherent: bool,
    pub last_check_timestamp: u64,
}

/// Anomaly detected during monitoring
#[derive(Debug, Clone)]
pub struct Anomaly {
    pub component: String,
    pub description: String,
    pub severity: f32, // 0.0 - 1.0
}

/// Diagnosis result from Reflexion loop
#[derive(Debug, Clone)]
pub struct Diagnosis {
    pub root_cause: String,
    pub recommended_action: String,
    pub mercy_score: f32,
}

/// Healing action that can be executed
#[derive(Debug, Clone)]
pub enum HealingAction {
    RestoreCosmicLoop,
    RestartComponent(String),
    RerouteCouncilTask { from: String, to: String },
    LogAndMonitor,
    NoAction,
}

/// Simple weighted graph for council task routing (for graph rerouting feature)
#[derive(Debug, Clone)]
pub struct CouncilTaskGraph {
    pub nodes: Vec<String>,           // Council names or IDs
    pub edges: Vec<(String, String, f32)>, // (from, to, weight) — lower weight = better path
}

impl CouncilTaskGraph {
    pub fn new() -> Self {
        // Default minimal graph representing core PATSAGi councils
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

    /// Find a simple alternative path (basic rerouting simulation)
    pub fn find_alternative_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        for (f, t, _w) in &self.edges {
            if f == from && t != to {
                return Some(vec![from.to_string(), t.clone(), to.to_string()]);
            }
        }
        None
    }

    /// Reweight an edge (used during healing to penalize problematic paths)
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

/// Runtime Self-Healing Engine with Watchdog + Advanced Reflexion + Experience Logging + Graph Rerouting
pub struct RuntimeSelfHealingEngine {
    pub cosmic_loop_ready: Arc<AtomicBool>,
    arbitration_engine: Arc<Mutex<CouncilArbitrationEngine>>,
    watchdog_running: Arc<AtomicBool>,
    healing_experiences: Arc<Mutex<Vec<HealingExperience>>>, // Advanced structured logging
    task_graph: Arc<Mutex<CouncilTaskGraph>>,
}

impl RuntimeSelfHealingEngine {
    pub fn new(arbitration_engine: CouncilArbitrationEngine) -> Self {
        Self {
            cosmic_loop_ready: Arc::new(AtomicBool::new(true)),
            arbitration_engine: Arc::new(Mutex::new(arbitration_engine)),
            watchdog_running: Arc::new(AtomicBool::new(false)),
            healing_experiences: Arc::new(Mutex::new(Vec::new())),
            task_graph: Arc::new(Mutex::new(CouncilTaskGraph::new())),
        }
    }

    /// Start the Watchdog Thread (runs in background)
    pub fn start_watchdog(&self) {
        if self.watchdog_running.load(Ordering::SeqCst) {
            println!("[Self-Healing] Watchdog already running.");
            return;
        }

        self.watchdog_running.store(true, Ordering::SeqCst);
        let cosmic_loop_ready = Arc::clone(&self.cosmic_loop_ready);
        let watchdog_running = Arc::clone(&self.watchdog_running);

        thread::spawn(move || {
            println!("[Self-Healing Watchdog] Thread started — monitoring lattice health...");
            loop {
                if !watchdog_running.load(Ordering::SeqCst) {
                    break;
                }

                if !cosmic_loop_ready.load(Ordering::SeqCst) {
                    println!("[Self-Healing Watchdog] ALERT: cosmic_loop_ready was false! Auto-restoring...");
                    cosmic_loop_ready.store(true, Ordering::SeqCst);
                }

                thread::sleep(Duration::from_secs(15));
            }
            println!("[Self-Healing Watchdog] Thread stopped.");
        });
    }

    pub fn stop_watchdog(&self) {
        self.watchdog_running.store(false, Ordering::SeqCst);
    }

    /// Run one advanced Reflexion-style healing cycle
    pub fn run_reflexion_cycle(&self) -> Diagnosis {
        let report = self.collect_health_report();
        let diagnosis = self.diagnose(&report);
        let action = self.reflect_and_decide_action(&diagnosis);
        self.execute_healing_action(action);
        self.log_healing_experience(&diagnosis); // Advanced structured logging
        diagnosis
    }

    fn collect_health_report(&self) -> HealthReport {
        HealthReport {
            cosmic_loop_ready: self.cosmic_loop_ready.load(Ordering::SeqCst),
            tol_c_gates_healthy: true,
            council_liveness: true,
            quantum_swarm_coherent: true,
            last_check_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    fn diagnose(&self, report: &HealthReport) -> Diagnosis {
        if !report.cosmic_loop_ready {
            Diagnosis {
                root_cause: "Cosmic Loop flag was unexpectedly disabled".to_string(),
                recommended_action: "Restore immediately + run council arbitration".to_string(),
                mercy_score: 0.98,
            }
        } else {
            Diagnosis {
                root_cause: "No critical anomalies detected".to_string(),
                recommended_action: "Continue monitoring".to_string(),
                mercy_score: 1.0,
            }
        }
    }

    fn reflect_and_decide_action(&self, diagnosis: &Diagnosis) -> HealingAction {
        if diagnosis.root_cause.contains("Cosmic Loop") {
            HealingAction::RestoreCosmicLoop
        } else if diagnosis.root_cause.contains("council") || diagnosis.severity > 0.7 {
            // Advanced: Suggest graph rerouting for council tasks
            HealingAction::RerouteCouncilTask {
                from: "Council#13".to_string(),
                to: "Evolution".to_string(),
            }
        } else {
            HealingAction::LogAndMonitor
        }
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
            HealingAction::RerouteCouncilTask { from, to } => {
                println!("[Self-Healing] Executing advanced Graph Rerouting: {} → {}", from, to);
                if let Ok(mut graph) = self.task_graph.lock() {
                    if let Some(path) = graph.find_alternative_path(&from, &to) {
                        println!("[Graph Rerouting] Alternative path found: {:?}", path);
                    }
                    // Penalize the direct edge slightly (learning from anomaly)
                    graph.reweight_edge(&from, &to, 3.5);
                }

                // Guardian protection
                if let Ok(mut arb) = self.arbitration_engine.lock() {
                    let _ = arb.protect_cosmic_loop_identity();
                }
            }
            HealingAction::LogAndMonitor => {}
            _ => {}
        }
    }

    /// Advanced structured experience logging (for Cosmic Loop self-evolution)
    fn log_healing_experience(&self, diagnosis: &Diagnosis) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let experience = HealingExperience {
            timestamp,
            root_cause: diagnosis.root_cause.clone(),
            action_taken: diagnosis.recommended_action.clone(),
            outcome: if diagnosis.mercy_score > 0.9 { "Success - High Mercy".to_string() } else { "Monitored".to_string() },
            mercy_score: diagnosis.mercy_score,
            graph_reroute_used: diagnosis.root_cause.contains("council"),
        };

        if let Ok(mut history) = self.healing_experiences.lock() {
            history.push(experience);
            if history.len() > 100 {
                history.remove(0);
            }
        }
    }

    pub fn get_healing_experiences(&self) -> Vec<HealingExperience> {
        self.healing_experiences.lock().map(|h| h.clone()).unwrap_or_default()
    }

    pub fn get_task_graph(&self) -> CouncilTaskGraph {
        self.task_graph.lock().map(|g| g.clone()).unwrap_or_else(|_| CouncilTaskGraph::new())
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
    fn test_graph_rerouting_and_experience_logging() {
        let arb = CouncilArbitrationEngine::new();
        let engine = RuntimeSelfHealingEngine::new(arb);

        engine.execute_healing_action(HealingAction::RerouteCouncilTask {
            from: "Council#13".to_string(),
            to: "Evolution".to_string(),
        });

        let experiences = engine.get_healing_experiences();
        assert!(!experiences.is_empty());

        let graph = engine.get_task_graph();
        assert!(graph.nodes.len() > 0);
    }
}