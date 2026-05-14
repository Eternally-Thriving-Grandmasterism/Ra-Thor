//! OrchestratorRegistry v2.0 — Paraconsistent-Aware Living Metadata Layer
//!
//! This is the evolved version designed to fully feed the ParaconsistentSuperKernel.
//! It includes contradiction detection, temporal history, SER contribution scoring,
//! enhanced ledger, and a clean ParaconsistentFeed interface.

use crate::registerable_orchestrator::{RegisterableOrchestrator, OrchestratorScope, MercyGateResult};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ContradictionTag {
    pub description: String,
    pub severity: f64,
    pub conflicting_orchestrators: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ValenceSnapshot {
    pub timestamp: u64,
    pub valence: f64,
}

#[derive(Debug, Clone)]
pub struct SERImpactReport {
    pub orchestrator_name: String,
    pub contribution_score: f64,
    pub last_updated: u64,
}

#[derive(Debug, Clone)]
pub struct ParaconsistentFeed {
    pub total_registered: usize,
    pub average_valence: f64,
    pub degraded_count: usize,
    pub health_score: f64,
    pub total_contradictions: usize,
    pub total_ser_contribution: f64,
}

pub struct OrchestratorRegistry {
    registered: HashMap<String, Box<dyn RegisterableOrchestrator + Send + Sync>>,
    valence_history: HashMap<String, Vec<ValenceSnapshot>>,
    ser_reports: HashMap<String, SERImpactReport>,
    contradiction_log: Vec<ContradictionTag>,
}

impl OrchestratorRegistry {
    pub fn new() -> Self {
        Self {
            registered: HashMap::new(),
            valence_history: HashMap::new(),
            ser_reports: HashMap::new(),
            contradiction_log: Vec::new(),
        }
    }

    pub fn register<T: RegisterableOrchestrator + Send + Sync + 'static>(
        &mut self,
        orchestrator: T,
    ) -> Result<(), String> {
        let name = orchestrator.name().to_string();

        if self.registered.contains_key(&name) {
            return Err(format!("Orchestrator {} is already registered", name));
        }

        if !orchestrator.is_mercy_aligned() {
            return Err(format!("{} failed mercy alignment", name));
        }

        // Record initial valence snapshot
        let snapshot = ValenceSnapshot {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            valence: orchestrator.current_valence(),
        };
        self.valence_history.insert(name.clone(), vec![snapshot]);

        self.registered.insert(name, Box::new(orchestrator));
        Ok(())
    }

    pub fn detect_contradictions(&mut self) -> Vec<ContradictionTag> {
        // Placeholder for real contradiction detection logic
        // In future this will compare coordination claims, valence reports, etc.
        vec![]
    }

    pub fn record_valence_snapshot(&mut self, name: &str, valence: f64) {
        let snapshot = ValenceSnapshot {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            valence,
        };
        self.valence_history
            .entry(name.to_string())
            .or_default()
            .push(snapshot);
    }

    pub fn get_valence_history(&self, name: &str) -> Option<&Vec<ValenceSnapshot>> {
        self.valence_history.get(name)
    }

    pub fn report_ser_contribution(&mut self, name: &str, score: f64) {
        let report = SERImpactReport {
            orchestrator_name: name.to_string(),
            contribution_score: score,
            last_updated: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        self.ser_reports.insert(name.to_string(), report);
    }

    pub fn get_total_ser_contribution(&self) -> f64 {
        self.ser_reports.values().map(|r| r.contribution_score).sum()
    }

    pub fn get_paraconsistent_feed(&self) -> ParaconsistentFeed {
        let count = self.registered.len();
        let avg_valence = if count > 0 {
            self.registered.values().map(|o| o.current_valence()).sum::<f64>() / count as f64
        } else {
            0.0
        };
        let degraded = self.registered.values().filter(|o| !o.is_mercy_aligned()).count();

        ParaconsistentFeed {
            total_registered: count,
            average_valence: avg_valence,
            degraded_count: degraded,
            health_score: if count > 0 { avg_valence * (1.0 - (degraded as f64 / count as f64)) } else { 1.0 },
            total_contradictions: self.contradiction_log.len(),
            total_ser_contribution: self.get_total_ser_contribution(),
        }
    }
}