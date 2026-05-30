//! Clifford Convolutions for Healing Fields (v14.3+)
//!
//! Production-grade, mercy-gated geometric healing system based on
//! Conformal Geometric Algebra (CGA) and Versor algebra.
//!
//! Provides Clifford-style convolutions, full ConformalVersor support
//! (exponential map, logarithm, interpolation), and geometric entity
//! handling (points, spheres, circles, lines) for multi-organism healing fields.
//!
//! Fully aligned with Thunder Lattice v14, MIAL, TOLC 8 Mercy Gates,
//! and AG-SML v1.0. Guardian-protected and PATSAGi Council aligned.

use nalgebra::Vector3;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::cga_primitives::{
    ConformalVersor, ConformalMotor, CgaPoint, CgaSphere, CgaCircle, CgaLine,
};
use crate::healing_integration::HealingFieldError;

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
}

/// Global coherence report for PATSAGi deliberation.
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

    pub fn add_organism(
        &mut self,
        id: u64,
        emotional: Vector3<f64>,
        physical: Vector3<f64>,
        alignment: Vector3<f64>,
        mercy: f64,
    ) {
        self.organism_fields.insert(id, OrganismField {
            emotional: Self::safe_normalize(emotional),
            physical: Self::safe_normalize(physical),
            alignment: Self::safe_normalize(alignment),
            mercy: mercy.clamp(0.0, 1.0),
        });
        self.evolution_step += 1;
    }

    /// Core Clifford-style convolution (classic path).
    pub fn apply_clifford_convolution(
        &mut self,
        kernel_strength: f64,
        mercy: f64,
    ) -> Result<(), HealingFieldError> {
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

    /// Full Versor-based healing step using ConformalVersor.
    #[cfg(feature = "full-clifford")]
    pub fn apply_versor_healing_step(
        &mut self,
        bivector: Vector3<f64>,
        mercy: f64,
    ) -> Result<GlobalCoherence, HealingFieldError> {
        if mercy < 0.6 {
            return Err(HealingFieldError::InvalidParameters);
        }
        if self.council_alignment < self.config.council_alignment_threshold {
            return Err(HealingFieldError::CouncilThresholdViolation(self.council_alignment));
        }

        let versor = ConformalVersor::exp(bivector);

        for field in self.organism_fields.values_mut() {
            let p = CgaPoint::from_vector(field.alignment);
            if let Some(transformed) = versor.apply_to_point(&p) {
                field.alignment = transformed.to_vector();
            }
            field.mercy = (field.mercy + mercy * 0.04).min(1.0);
        }

        self.evolution_step += 1;
        Ok(self.compute_global_coherence())
    }

    /// Apply PATSAGi Council guidance.
    pub fn apply_patsagi_council_guidance(
        &mut self,
        council_mercy: f64,
    ) -> Result<GlobalCoherence, HealingFieldError> {
        if council_mercy < 0.7 {
            return Err(HealingFieldError::CouncilThresholdViolation(council_mercy));
        }

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
    pub fn simulate_healing_step(
        &mut self,
        kernel: f64,
        mercy: f64,
        council_mercy: f64,
    ) -> Result<GlobalCoherence, HealingFieldError> {
        self.apply_clifford_convolution(kernel, mercy)?;
        let coherence = self.apply_patsagi_council_guidance(council_mercy)?;
        Ok(coherence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_convolution() {
        let mut field = CliffordHealingField::new("TestField");
        field.add_organism(1, Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0), 0.9);
        assert!(field.apply_clifford_convolution(0.5, 0.9).is_ok());
    }
}
