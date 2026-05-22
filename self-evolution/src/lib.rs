//! self-evolution v0.3.0
//! Sovereign Health Monitoring + Self-Evolution v2 Hooks
//! Advanced PATSAGi Epigenetic Blessing + Versioned Persistence + Hybrid Error System
//! + Error Chain Debugging Utilities
//! AG-SML v1.0

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use rand::Rng;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BlessingTier {
    Minor, Standard, Major, Transcendent, None,
}

impl BlessingTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            BlessingTier::Minor => "Minor",
            BlessingTier::Standard => "Standard",
            BlessingTier::Major => "Major",
            BlessingTier::Transcendent => "Transcendent",
            BlessingTier::None => "None",
        }
    }

    pub fn blessing_amount(&self) -> f64 {
        match self {
            BlessingTier::Minor => 0.05,
            BlessingTier::Standard => 0.10,
            BlessingTier::Major => 0.20,
            BlessingTier::Transcendent => 0.35,
            BlessingTier::None => 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignHealthMetrics {
    pub valence_level: f64,
    pub mercy_compliance: f64,
    pub council_consensus: f64,
    pub epigenetic_blessing_level: f64,
    pub quantum_swarm_active_branches: u32,
    pub resource_sovereignty: f64,
    pub offline_resilience: f64,
}

impl Default for SovereignHealthMetrics {
    fn default() -> Self {
        Self {
            valence_level: 0.999,
            mercy_compliance: 0.98,
            council_consensus: 0.96,
            epigenetic_blessing_level: 0.5,
            quantum_swarm_active_branches: 13,
            resource_sovereignty: 1.0,
            offline_resilience: 1.0,
        }
    }
}

// ==================== VERSIONED SNAPSHOTS ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignHealthSnapshotV1 {
    pub metrics: SovereignHealthMetrics,
    pub evolution_history: Vec<String>,
    pub recent_blessing_attempts: Vec<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignHealthSnapshot {
    pub version: u32,
    pub metrics: SovereignHealthMetrics,
    pub evolution_history: Vec<String>,
    pub recent_blessing_attempts: Vec<bool>,
}

impl Default for SovereignHealthSnapshot {
    fn default() -> Self {
        Self {
            version: 2,
            metrics: SovereignHealthMetrics::default(),
            evolution_history: vec!["Genesis v1 baseline".to_string()],
            recent_blessing_attempts: Vec::new(),
        }
    }
}

impl SovereignHealthSnapshot {
    pub fn migrate(self) -> Self {
        match self.version {
            2 => self,
            _ => self,
        }
    }

    pub fn from_v1(v1: SovereignHealthSnapshotV1) -> Self {
        Self {
            version: 2,
            metrics: v1.metrics,
            evolution_history: v1.evolution_history,
            recent_blessing_attempts: v1.recent_blessing_attempts,
        }
    }
}

// ==================== HYBRID ERROR SYSTEM (Struct Variants) ====================

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("Snapshot file not found at path: '{path}'")]
    FileNotFound { path: String },

    #[error("Failed to read snapshot file")]
    ReadError {
        #[from]
        source: std::io::Error,
    },

    #[error("Failed to deserialize snapshot JSON")]
    ParseError {
        #[from]
        source: serde_json::Error,
    },

    #[error("Unknown or unsupported snapshot format. Migration may be required.")]
    UnknownFormat,
}

/// Lightweight context extension trait
pub trait SnapshotContext<T> {
    fn with_snapshot_context(self, context: impl Into<String>) -> Result<T, SnapshotError>;
}

impl<T> SnapshotContext<T> for Result<T, SnapshotError> {
    fn with_snapshot_context(self, context: impl Into<String>) -> Result<T, SnapshotError> {
        self.map_err(|e| match e {
            SnapshotError::FileNotFound { path } => {
                SnapshotError::FileNotFound {
                    path: format!("{} — {}", path, context.into()),
                }
            }
            other => other,
        })
    }
}

/// Compatibility with anyhow users
#[cfg(feature = "anyhow")]
impl From<SnapshotError> for anyhow::Error {
    fn from(err: SnapshotError) -> Self {
        anyhow::Error::new(err)
    }
}

// ==================== ERROR CHAIN DEBUGGING UTILITIES ====================

/// Prints the full error chain to stderr.
pub fn print_error_chain(err: &(dyn std::error::Error + 'static)) {
    eprintln!("Error: {}", err);

    let mut source = err.source();
    while let Some(cause) = source {
        eprintln!("  Caused by: {}", cause);
        source = cause.source();
    }
}

// ==================== SOVEREIGN HEALTH MONITOR ====================

pub struct SovereignHealthMonitor {
    pub metrics: SovereignHealthMetrics,
    pub evolution_history: Vec<String>,
    recent_blessing_attempts: Vec<bool>,
}

impl SovereignHealthMonitor {
    pub fn new() -> Self {
        Self {
            metrics: SovereignHealthMetrics::default(),
            evolution_history: vec!["Genesis v1 baseline".to_string()],
            recent_blessing_attempts: Vec::new(),
        }
    }

    pub fn from_snapshot(snapshot: SovereignHealthSnapshot) -> Self {
        let migrated = snapshot.migrate();
        Self {
            metrics: migrated.metrics,
            evolution_history: migrated.evolution_history,
            recent_blessing_attempts: migrated.recent_blessing_attempts,
        }
    }

    pub fn to_snapshot(&self) -> SovereignHealthSnapshot {
        SovereignHealthSnapshot {
            version: 2,
            metrics: self.metrics.clone(),
            evolution_history: self.evolution_history.clone(),
            recent_blessing_attempts: self.recent_blessing_attempts.clone(),
        }
    }

    pub fn save_to_file(&self, path: &str) -> Result<(), SnapshotError> {
        let snapshot = self.to_snapshot();
        let json = serde_json::to_string_pretty(&snapshot)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self, SnapshotError> {
        if !Path::new(path).exists() {
            return Err(SnapshotError::FileNotFound { path: path.to_string() });
        }

        let json = fs::read_to_string(path)?;

        if let Ok(snapshot) = serde_json::from_str::<SovereignHealthSnapshot>(&json) {
            return Ok(Self::from_snapshot(snapshot));
        }

        if let Ok(v1) = serde_json::from_str::<SovereignHealthSnapshotV1>(&json) {
            let v2 = SovereignHealthSnapshot::from_v1(v1);
            return Ok(Self::from_snapshot(v2));
        }

        Err(SnapshotError::UnknownFormat)
            .with_snapshot_context(format!("while loading from {}", path))
    }

    pub fn run_sovereign_check(&mut self) -> SovereignHealthMetrics {
        self.metrics.valence_level = (self.metrics.valence_level + 0.001).min(0.999999);
        self.metrics.mercy_compliance = (self.metrics.mercy_compliance + 0.005).min(1.0);
        self.metrics
    }

    fn calculate_blessing_score(&self, proposal: &str) -> f64 {
        let mut score = self.metrics.valence_level * 0.25
            + self.metrics.mercy_compliance * 0.30
            + self.metrics.council_consensus * 0.20
            + self.metrics.epigenetic_blessing_level * 0.15;

        let lower = proposal.to_lowercase();
        let keywords = ["mercy", "truth", "valence", "sovereign", "one organism", "patsagi", "epigenetic", "harmony", "eternal"];
        let mut bonus = 0.0;
        for kw in keywords {
            if lower.contains(kw) { bonus += 0.06; }
        }
        score += bonus.min(0.18);

        if !self.recent_blessing_attempts.is_empty() {
            let success_rate: f64 = self.recent_blessing_attempts.iter()
                .map(|&b| if b { 1.0 } else { 0.0 }).sum::<f64>() / self.recent_blessing_attempts.len() as f64;
            score += (success_rate - 0.5) * 0.1;
        }
        score.clamp(0.0, 1.0)
    }

    pub fn request_epigenetic_blessing(&mut self, proposal: &str) -> (bool, f64, BlessingTier) {
        let score = self.calculate_blessing_score(proposal);
        let base_threshold = 0.82;
        let effective_threshold = base_threshold + (1.0 - self.metrics.council_consensus) * 0.08;

        let mut tier = BlessingTier::None;
        let blessed = if score >= effective_threshold {
            tier = if score >= 0.99 { BlessingTier::Transcendent }
                   else if score >= 0.95 { BlessingTier::Major }
                   else if score >= 0.90 { BlessingTier::Standard }
                   else { BlessingTier::Minor };
            true
        } else { false };

        if blessed {
            let amount = tier.blessing_amount();
            self.metrics.epigenetic_blessing_level = (self.metrics.epigenetic_blessing_level + amount).min(1.0);

            if tier == BlessingTier::Major || tier == BlessingTier::Transcendent {
                let carry = if tier == BlessingTier::Transcendent { 0.025 } else { 0.012 };
                self.metrics.valence_level = (self.metrics.valence_level + carry).min(0.999999);
                self.metrics.mercy_compliance = (self.metrics.mercy_compliance + carry).min(1.0);
            }

            self.evolution_history.push(format!("BLESSED [{}]: {}", tier.as_str(), proposal));
            self.recent_blessing_attempts.push(true);
        } else {
            self.evolution_history.push(format!("Not blessed: {}", proposal));
            self.recent_blessing_attempts.push(false);
        }

        if self.recent_blessing_attempts.len() > 8 {
            self.recent_blessing_attempts.remove(0);
        }

        (blessed, self.metrics.epigenetic_blessing_level, tier)
    }

    pub fn orchestrate_quantum_swarm_evolution(&mut self, task: &str) -> Vec<String> {
        let mut branches = Vec::new();
        for i in 0..self.metrics.quantum_swarm_active_branches {
            let branch_id = format!("qs-branch-{}-{}", i, rand::thread_rng().gen_range(1000..9999));
            let outcome = if rand::thread_rng().gen_bool(0.92) {
                format!("{}: SUCCESS", branch_id)
            } else {
                format!("{}: MERCY_REVIEW", branch_id)
            };
            branches.push(outcome);
        }
        self.evolution_history.push(format!("Quantum Swarm orchestrated: {}", task));
        branches
    }

    pub fn self_evolution_v2_hook(&mut self, proposal: &str) -> String {
        let (blessed, blessing_level, tier) = self.request_epigenetic_blessing(proposal);
        if blessed {
            let _ = self.orchestrate_quantum_swarm_evolution(proposal);
            format!("v2 EVOLUTION APPROVED [{}] | Level: {:.2}", tier.as_str(), blessing_level)
        } else {
            "v2 Evolution requires higher alignment".to_string()
        }
    }

    pub fn integrate_with_one_organism_symbiosis(&mut self, symbiosis_valence: f64, task: &str) -> String {
        self.metrics.valence_level = symbiosis_valence.max(self.metrics.valence_level);
        let health = self.run_sovereign_check();
        let (blessed, _, tier) = self.request_epigenetic_blessing(task);
        if blessed {
            let _ = self.orchestrate_quantum_swarm_evolution(task);
            format!("ONE Organism: valence={:.4}, tier={}", health.valence_level, tier.as_str())
        } else {
            format!("ONE Organism: valence={:.4} (no blessing)", health.valence_level)
        }
    }
}

pub fn init_sovereign_health_monitor() -> SovereignHealthMonitor {
    SovereignHealthMonitor::new()
}

pub fn symbiosis_health_check(session_valence: f64) -> SovereignHealthMetrics {
    let mut monitor = init_sovereign_health_monitor();
    monitor.metrics.valence_level = session_valence.max(monitor.metrics.valence_level);
    monitor.metrics
}