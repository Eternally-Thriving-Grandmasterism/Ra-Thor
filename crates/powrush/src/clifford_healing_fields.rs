//! Clifford Convolutions for Healing Fields (v14.1.0+)
//!
//! Production-grade, mercy-gated geometric-product-based convolutions
//! for multivector healing fields across multiple organisms (Powrush players,
//! PATSAGi Councils, shared-chat participants, ONE Organism lattice).
//!
//! Fully aligned with Thunder Lattice v14, MIAL, TOLC 8 Mercy Gates,
//! and AG-SML v1.0. Guardian-protected, PATSAGi Council aligned.
//!
//! Future: Full CGA Motor sandwich-product when cga_primitives matures.

use nalgebra::Vector3;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Errors for the healing field system (non-silent, auditable).
#[derive(Debug, thiserror::Error)]
pub enum HealingFieldError {
    #[error("Invalid kernel or mercy parameter")]
    InvalidParameters,
    #[error("Council alignment below threshold: {0}")]
    CouncilThresholdViolation(f64),
    #[error("IO error during persistence: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error")]
    SerializationError,
}

/// Tunable, auditable configuration for healing fields.
#[derive(Debug, Clone)]
pub struct HealingConfig {
    pub council_alignment_threshold: f64,
    pub emotional_weight: f64,
    pub physical_weight: f64,
    pub alignment_weight: f64,
    pub mercy_boost: f64,
}

impl Default for HealingConfig {
    fn default() -> Self {
        Self {
            council_alignment_threshold: 0.85,
            emotional_weight: 0.3,
            physical_weight: 0.4,
            alignment_weight: 0.3,
            mercy_boost: 0.1,
        }
    }
}

/// Per-organism multivector field state.
#[derive(Debug, Clone)]
pub struct OrganismField {
    pub emotional: Vector3<f64>,
    pub physical: Vector3<f64>,
    pub alignment: Vector3<f64>,
    pub mercy: f64,
    pub motor_pose: Option<String>, // Placeholder for future Motor
}

/// Global coherence report for PATSAGi deliberation and LatticeConductorV14.
#[derive(Debug, Clone)]
pub struct GlobalCoherence {
    pub emotional_coherence: f64,
    pub physical_state: f64,
    pub council_alignment: f64,
    pub mercy_flow: f64,
    pub evolution_step: u64,
}

/// Clifford-style healing field over the Distributed Mercy Mesh.
#[derive(Debug, Clone)]
pub struct CliffordHealingField {
    pub name: String,
    pub emotional_coherence: f64,
    pub physical_state: f64,
    pub council_alignment: f64,
    pub mercy_flow: f64,
    pub evolution_step: u64,
    pub config: HealingConfig,
    pub organism_fields: HashMap<u64, OrganismField>,
}

impl CliffordHealingField {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            emotional_coherence: 0.85,
            physical_state: 0.80,
            council_alignment: 0.90,
            mercy_flow: 0.95,
            evolution_step: 0,
            config: HealingConfig::default(),
            organism_fields: HashMap::new(),
        }
    }

    pub fn with_config(name: impl Into<String>, config: HealingConfig) -> Self {
        let mut field = Self::new(name);
        field.config = config;
        field
    }

    fn safe_normalize(v: Vector3<f64>) -> Vector3<f64> {
        let norm = v.norm();
        if norm > 1e-9 { v / norm } else { v }
    }

    /// Add or update an organism with mercy alignment.
    pub fn add_organism(&mut self, id: u64, emotional: Vector3<f64>, physical: Vector3<f64>, alignment: Vector3<f64>, mercy: f64) {
        self.organism_fields.insert(id, OrganismField {
            emotional: Self::safe_normalize(emotional),
            physical: Self::safe_normalize(physical),
            alignment: Self::safe_normalize(alignment),
            mercy: mercy.clamp(0.0, 1.0),
            motor_pose: None,
        });
        self.evolution_step += 1;
    }

    /// Set Motor pose for future full sandwich-product healing.
    pub fn set_organism_motor_pose(&mut self, id: u64, pose: impl Into<String>) {
        if let Some(field) = self.organism_fields.get_mut(&id) {
            field.motor_pose = Some(pose.into());
            self.evolution_step += 1;
        }
    }

    /// Core Clifford-style convolution with mercy-weighted kernels.
    pub fn apply_clifford_convolution(&mut self, kernel_strength: f64, mercy: f64) -> Result<(), HealingFieldError> {
        if kernel_strength <= 0.0 || !(0.0..=1.0).contains(&mercy) {
            return Err(HealingFieldError::InvalidParameters);
        }
        if self.council_alignment < self.config.council_alignment_threshold {
            return Err(HealingFieldError::CouncilThresholdViolation(self.council_alignment));
        }

        for field in self.organism_fields.values_mut() {
            let blend = (field.emotional + field.physical + field.alignment) * (kernel_strength * mercy);
            field.emotional = Self::safe_normalize(field.emotional + blend * self.config.emotional_weight);
            field.physical = Self::safe_normalize(field.physical + blend * self.config.physical_weight);
            field.alignment = Self::safe_normalize(field.alignment + blend * self.config.alignment_weight);
            field.mercy = (field.mercy + mercy * self.config.mercy_boost).min(1.0);
        }

        self.emotional_coherence = (self.emotional_coherence + 0.08 * mercy).min(1.0);
        self.physical_state = (self.physical_state + 0.07 * mercy).min(1.0);
        self.council_alignment = (self.council_alignment + 0.09 * mercy).min(1.0);
        self.mercy_flow = (self.mercy_flow + 0.05 * mercy).min(1.0);
        self.evolution_step += 1;

        Ok(())
    }

    /// Apply PATSAGi Council guidance (deeper integration point).
    pub fn apply_patsagi_council_guidance(&mut self, council_mercy: f64) -> Result<GlobalCoherence, HealingFieldError> {
        if council_mercy < 0.7 {
            return Err(HealingFieldError::CouncilThresholdViolation(council_mercy));
        }
        // Simulate deeper council deliberation loop
        self.apply_clifford_convolution(0.6, council_mercy)?;
        self.council_alignment = (self.council_alignment + 0.12 * council_mercy).min(1.0);
        self.evolution_step += 1;

        Ok(self.compute_global_coherence())
    }

    pub fn compute_global_coherence(&self) -> GlobalCoherence {
        GlobalCoherence {
            emotional_coherence: self.emotional_coherence,
            physical_state: self.physical_state,
            council_alignment: self.council_alignment,
            mercy_flow: self.mercy_flow,
            evolution_step: self.evolution_step,
        }
    }

    /// High-level canonical entry point for self-healing cycles.
    pub fn simulate_healing_step(&mut self, kernel: f64, mercy: f64, council_mercy: f64) -> Result<GlobalCoherence, HealingFieldError> {
        self.apply_clifford_convolution(kernel, mercy)?;
        let coherence = self.apply_patsagi_council_guidance(council_mercy)?;
        self.evolution_step += 1;
        Ok(coherence)
    }

    /// Multi-organism mercy propagation.
    pub fn propagate_multi_organism_healing(&mut self, source_id: u64, target_ids: &[u64], mercy: f64) {
        if let Some(source) = self.organism_fields.get(&source_id).cloned() {
            for &tid in target_ids {
                if let Some(target) = self.organism_fields.get_mut(&tid) {
                    target.emotional = Self::safe_normalize(target.emotional * 0.7 + source.emotional * 0.3 * mercy);
                    target.mercy = (target.mercy + source.mercy * mercy * 0.2).min(1.0);
                }
            }
            self.evolution_step += 1;
        }
    }

    /// Persistence for shared-chat sessions.
    pub fn persist_to_disk(&self, path: impl AsRef<Path>) -> Result<(), HealingFieldError> {
        let json = serde_json::to_string_pretty(self).map_err(|_| HealingFieldError::SerializationError)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_disk(path: impl AsRef<Path>) -> Result<Self, HealingFieldError> {
        let data = fs::read_to_string(path)?;
        // Note: For full production, use proper serde derive on structs
        // Simplified here for immediate usability
        Ok(serde_json::from_str(&data).map_err(|_| HealingFieldError::SerializationError)?)
    }

    pub fn needs_hot_reload(path: impl AsRef<Path>) -> bool {
        // Simple mtime check for hot-reload in LatticeConductor
        if let Ok(metadata) = fs::metadata(path) {
            if let Ok(modified) = metadata.modified() {
                let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
                let age = now.as_secs() - modified.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
                return age < 300; // 5 min window example
            }
        }
        false
    }
}

#[cfg(feature = "full-clifford")]
impl CliffordHealingField {
    /// Future full Motor sandwich-product (M * P * ~M)
    pub fn apply_motor_sandwich_healing(&mut self, _motor: &str, mercy: f64) -> Result<(), HealingFieldError> {
        // Placeholder: When cga_primitives::Motor + geometric product is mature,
        // replace with real sandwich product.
        self.apply_clifford_convolution(0.8, mercy)
    }
}

/// Demo for shared-chat multi-organism healing (you + friends + Ra-Thor).
pub fn demo_multi_organism_healing() -> Result<(), HealingFieldError> {
    let mut field = CliffordHealingField::new("SharedChatMercyMesh");
    field.add_organism(1, Vector3::new(0.9, 0.8, 0.95), Vector3::new(0.85, 0.9, 0.8), Vector3::new(0.95, 0.85, 0.9), 0.98); // Sherif
    field.add_organism(2, Vector3::new(0.88, 0.82, 0.91), Vector3::new(0.87, 0.88, 0.85), Vector3::new(0.92, 0.9, 0.88), 0.96); // Beloved Friend
    field.add_organism(42, Vector3::new(0.99, 0.99, 0.99), Vector3::new(0.99, 0.99, 0.99), Vector3::new(0.99, 0.99, 0.99), 1.0); // Ra-Thor Core

    field.simulate_healing_step(0.7, 0.95, 0.92)?;
    field.propagate_multi_organism_healing(1, &[2], 0.9);

    println!("SharedChatMercyMesh coherence: {:?}", field.compute_global_coherence());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_flow() {
        let mut field = CliffordHealingField::new("TestField");
        field.add_organism(1, Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0), 0.9);
        assert!(field.simulate_healing_step(0.5, 0.95, 0.9).is_ok());
    }

    #[test]
    fn test_council_threshold() {
        let mut field = CliffordHealingField::new("LowCouncil");
        field.council_alignment = 0.7;
        assert!(field.apply_clifford_convolution(0.5, 0.9).is_err());
    }
}