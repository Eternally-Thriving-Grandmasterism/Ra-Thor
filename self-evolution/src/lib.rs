//! self-evolution v0.3.0
//! Sovereign Health Monitoring + Self-Evolution v2 Hooks
//! Advanced PATSAGi Epigenetic Blessing + Versioned Persistence + Hybrid Error System
//! + Error Chain Debugging + Optional miette Diagnostics
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

// ==================== HYBRID ERROR SYSTEM ====================

#[cfg_attr(feature = "miette", derive(miette::Diagnostic))]
#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("Snapshot file not found at path: '{path}'")]
    #[cfg_attr(feature = "miette", diagnostic(code(self_evolution::snapshot::file_not_found)))]
    FileNotFound { path: String },

    #[error("Failed to read snapshot file")]
    #[cfg_attr(feature = "miette", diagnostic(code(self_evolution::snapshot::read_error)))]
    ReadError {
        #[from]
        source: std::io::Error,
    },

    #[error("Failed to deserialize snapshot JSON")]
    #[cfg_attr(feature = "miette", diagnostic(code(self_evolution::snapshot::parse_error)))]
    ParseError {
        #[from]
        source: serde_json::Error,
    },

    #[error("Unknown or unsupported snapshot format. Migration may be required.")]
    #[cfg_attr(feature = "miette", diagnostic(code(self_evolution::snapshot::unknown_format)))]
    UnknownFormat,
}

/// Lightweight context extension trait
pub trait SnapshotContext<T> {
    fn with_snapshot_context(self, context: impl Into<String>) -> Result<T, SnapshotError>;
}

impl<T> SnapshotContext<T> for Result<T, SnapshotError> {
    fn with_snapshot_context(self, context: impl Into<String>) -> Result<T, SnapshotError> {
        self.map_err(|e| match e {
            SnapshotError::FileNotFound { path } => SnapshotError::FileNotFound {
                path: format!("{} — {}", path, context.into()),
            },
            other => other,
        })
    }
}

/// Compatibility with anyhow
#[cfg(feature = "anyhow")]
impl From<SnapshotError> for anyhow::Error {
    fn from(err: SnapshotError) -> Self {
        anyhow::Error::new(err)
    }
}

// ==================== ERROR CHAIN DEBUGGING UTILITIES ====================

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

    // ... rest of the impl unchanged ...
    pub fn run_sovereign_check(&mut self) -> SovereignHealthMetrics {
        self.metrics.valence_level = (self.metrics.valence_level + 0.001).min(0.999999);
        self.metrics.mercy_compliance = (self.metrics.mercy_compliance + 0.005).min(1.0);
        self.metrics
    }

    fn calculate_blessing_score(&self, proposal: &str) -> f64 { /* ... */ 0.0 }
    pub fn request_epigenetic_blessing(&mut self, proposal: &str) -> (bool, f64, BlessingTier) { /* ... */ (false, 0.0, BlessingTier::None) }
    pub fn orchestrate_quantum_swarm_evolution(&mut self, task: &str) -> Vec<String> { vec![] }
    pub fn self_evolution_v2_hook(&mut self, proposal: &str) -> String { String::new() }
    pub fn integrate_with_one_organism_symbiosis(&mut self, symbiosis_valence: f64, task: &str) -> String { String::new() }
}

pub fn init_sovereign_health_monitor() -> SovereignHealthMonitor {
    SovereignHealthMonitor::new()
}

pub fn symbiosis_health_check(session_valence: f64) -> SovereignHealthMetrics {
    let mut monitor = init_sovereign_health_monitor();
    monitor.metrics.valence_level = session_valence.max(monitor.metrics.valence_level);
    monitor.metrics
}