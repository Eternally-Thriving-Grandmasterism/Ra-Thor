//! Clifford Convolutions for Healing Fields (v14.8.2)
//! Minimal complete production surface for Lattice Conductor v14.
//! Full geometric algebra expansion reserved for `full-clifford` feature.

use nalgebra::Vector3;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, thiserror::Error)]
pub enum HealingFieldError {
    #[error("Mercy gate violated: {0}")]
    MercyGateViolation(String),
    #[error("Invalid organism id")]
    InvalidOrganism,
    #[error("Persistence error: {0}")]
    Persistence(String),
}

#[derive(Debug, Clone)]
pub struct HealingConfig {
    pub min_mercy: f64,
    pub max_organisms: usize,
    pub enable_motor_sandwich: bool,
}

impl Default for HealingConfig {
    fn default() -> Self {
        Self {
            min_mercy: 0.7,
            max_organisms: 256,
            enable_motor_sandwich: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlobalCoherence {
    pub average_mercy: f64,
    pub organism_count: usize,
    pub evolution_step: u64,
}

#[derive(Debug, Clone)]
pub struct OrganismField {
    pub id: u64,
    pub name: String,
    pub emotional: Vector3<f64>,
    pub mercy: f64,
    pub coherence: f64,
}

#[derive(Debug, Clone)]
pub struct Motor {
    pub scalar: f64,
    pub vector: Vector3<f64>,
}

impl Motor {
    pub fn new(scalar: f64, vector: Vector3<f64>) -> Self {
        Self { scalar, vector }
    }

    pub fn reverse(&self) -> Self {
        Self {
            scalar: self.scalar,
            vector: -self.vector,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CliffordHealingField {
    pub name: String,
    pub organism_fields: HashMap<u64, OrganismField>,
    pub evolution_step: u64,
    pub config: HealingConfig,
}

impl CliffordHealingField {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            organism_fields: HashMap::new(),
            evolution_step: 0,
            config: HealingConfig::default(),
        }
    }

    pub fn add_organism(&mut self, id: u64, name: impl Into<String>, coherence: f64) {
        if self.organism_fields.len() >= self.config.max_organisms {
            return;
        }
        self.organism_fields.insert(
            id,
            OrganismField {
                id,
                name: name.into(),
                emotional: Vector3::new(0.0, 0.0, coherence),
                mercy: coherence.clamp(0.0, 1.0),
                coherence: coherence.clamp(0.0, 1.0),
            },
        );
    }

    pub fn apply_patsagi_council_guidance(&mut self, mercy: f64, coherence_boost: f64) {
        for field in self.organism_fields.values_mut() {
            field.mercy = (field.mercy * 0.85 + mercy * 0.15).clamp(0.0, 1.0);
            field.coherence = (field.coherence * 0.8 + coherence_boost * 0.2).clamp(0.0, 1.0);
        }
        self.evolution_step += 1;
    }

    pub fn simulate_healing_step(&mut self, mercy: f64) -> Result<GlobalCoherence, HealingFieldError> {
        if mercy < self.config.min_mercy {
            return Err(HealingFieldError::MercyGateViolation(format!(
                "mercy {mercy:.3} below minimum {:.3}",
                self.config.min_mercy
            )));
        }
        self.apply_patsagi_council_guidance(mercy, mercy);
        Ok(self.global_coherence())
    }

    pub fn global_coherence(&self) -> GlobalCoherence {
        let count = self.organism_fields.len();
        let average_mercy = if count == 0 {
            1.0
        } else {
            self.organism_fields.values().map(|o| o.mercy).sum::<f64>() / count as f64
        };
        GlobalCoherence {
            average_mercy,
            organism_count: count,
            evolution_step: self.evolution_step,
        }
    }

    pub fn persist_to_disk(&self, _path: &Path) {
        // Lightweight no-op persistence hook — full serialization in later iteration
        println!(
            "[CliffordHealingField] persist_to_disk called for '{}' ({} organisms, step {})",
            self.name,
            self.organism_fields.len(),
            self.evolution_step
        );
    }

    #[cfg(feature = "full-clifford")]
    pub fn apply_motor_sandwich_healing(
        &mut self,
        _source_id: u64,
        target_ids: &[u64],
        motor: &Motor,
        mercy: f64,
    ) -> Result<(), HealingFieldError> {
        if mercy < self.config.min_mercy {
            return Err(HealingFieldError::MercyGateViolation(
                "Insufficient mercy for sandwich".into(),
            ));
        }
        for &tid in target_ids {
            if let Some(target) = self.organism_fields.get_mut(&tid) {
                let scale = motor.scalar * mercy;
                target.emotional = (target.emotional * scale + motor.vector * mercy).normalize();
                target.mercy = (target.mercy + mercy * 0.15).min(1.0);
            }
        }
        self.evolution_step += 1;
        Ok(())
    }
}
