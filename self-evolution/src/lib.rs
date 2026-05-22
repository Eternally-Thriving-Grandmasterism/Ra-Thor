//! self-evolution v0.3.0
//! Sovereign Health Monitoring + Self-Evolution v2 Hooks
//! Versioned Snapshot + Migration System (Robust + Future-proof)
//! AG-SML v1.0

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BlessingTier {
    Minor, Standard, Major, Transcendent, None,
}

impl BlessingTier {
    pub fn as_str(&self) -> &'static str { /* ... */ match self { BlessingTier::Minor => "Minor", BlessingTier::Standard => "Standard", BlessingTier::Major => "Major", BlessingTier::Transcendent => "Transcendent", BlessingTier::None => "None" } }
    pub fn blessing_amount(&self) -> f64 { /* ... */ match self { BlessingTier::Minor => 0.05, BlessingTier::Standard => 0.10, BlessingTier::Major => 0.20, BlessingTier::Transcendent => 0.35, BlessingTier::None => 0.0 } }
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

// ==================== VERSIONED SNAPSHOTS ====================

/// Version 1 Snapshot (original, for migration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignHealthSnapshotV1 {
    pub metrics: SovereignHealthMetrics,
    pub evolution_history: Vec<String>,
    pub recent_blessing_attempts: Vec<bool>,
}

/// Current Snapshot (Version 2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignHealthSnapshot {
    pub version: u32,
    pub metrics: SovereignHealthMetrics,
    pub evolution_history: Vec<String>,
    pub recent_blessing_attempts: Vec<bool>,
    // Future fields can be added here with #[serde(default)]
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
    /// Migrate from older versions to latest
    pub fn migrate(self) -> Self {
        match self.version {
            2 => self, // Already latest
            // Add future migration paths here, e.g.:
            // 1 => migrate_v1_to_v2(self),
            _ => {
                // Unknown version — try to treat as V2 or fail gracefully
                println!("Warning: Unknown snapshot version {}", self.version);
                self
            }
        }
    }

    /// Convert from V1 to current V2
    pub fn from_v1(v1: SovereignHealthSnapshotV1) -> Self {
        Self {
            version: 2,
            metrics: v1.metrics,
            evolution_history: v1.evolution_history,
            recent_blessing_attempts: v1.recent_blessing_attempts,
        }
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

    /// Save with automatic versioning
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let snapshot = self.to_snapshot();
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| format!("Serialization error: {}", e))?;
        fs::write(path, json).map_err(|e| format!("Write error: {}", e))?
        Ok(())
    }

    /// Load with automatic migration from older versions
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        if !Path::new(path).exists() {
            return Err(format!("File not found: {}", path));
        }

        let json = fs::read_to_string(path)
            .map_err(|e| format!("Read error: {}", e))?;

        // Try to detect version
        if let Ok(v2) = serde_json::from_str::<SovereignHealthSnapshot>(&json) {
            return Ok(Self::from_snapshot(v2));
        }

        // Fallback: try V1 migration
        if let Ok(v1) = serde_json::from_str::<SovereignHealthSnapshotV1>(&json) {
            let v2 = SovereignHealthSnapshot::from_v1(v1);
            return Ok(Self::from_snapshot(v2));
        }

        Err("Failed to parse snapshot (unknown format)".to_string())
    }

    // ... rest of the methods (request_epigenetic_blessing, etc.) remain the same ...
    pub fn run_sovereign_check(&mut self) -> SovereignHealthMetrics { /* ... */ self.metrics }
    // (Other methods unchanged for this update)
}

pub fn init_sovereign_health_monitor() -> SovereignHealthMonitor {
    SovereignHealthMonitor::new()
}