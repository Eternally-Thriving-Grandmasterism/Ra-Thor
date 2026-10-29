//! # Clifford Convolutions for Healing Fields (v14.3+)
//!
//! Production-grade, mercy-gated geometric healing system based on
//! Conformal Geometric Algebra (CGA) and Versor algebra.
//!
//! This module is a core component of **Powrush RBE**'s mercy infrastructure.
//! It enables organism-level and planetary-scale mercy propagation through
//! geometric transformations.
//!
//! ## Key Capabilities
//! - Tunable `HealingConfig` for emotional / physical / alignment weights
//! - `CliffordHealingField` managing per-organism multivector states
//! - Classic Clifford convolutions + full `ConformalVersor` healing steps
//! - PATSAGi Council guidance with mercy gating
//! - Global coherence computation for higher council deliberation
//!
//! ## Design Principles
//! - **Mercy-gated at every layer** (Layer 0 council checks)
//! - Privacy-by-design (no PII in geometric states)
//! - Pluggable normalization strategies
//! - Full test coverage
//! - AG-SML v1.0 licensed, PATSAGi aligned
//!
//! Builds on PR #189 Clifford planning and PR #190 stabilization.
//! Future: Deep integration with Real Estate Lattice for spatial mercy propagation.

use nalgebra::Vector3;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::cga_primitives::{
    ConformalVersor, ConformalMotor, CgaPoint, CgaSphere, CgaCircle, CgaLine,
};
use crate::healing_integration::HealingFieldError;

/// Tunable, auditable configuration for healing fields.
/// All weights and thresholds are exposed for PATSAGi Council review.
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
/// Represents emotional, physical, and alignment vectors + current mercy level.
#[derive(Debug, Clone)]
pub struct OrganismField {
    pub emotional: Vector3<f64>,
    pub physical: Vector3<f64>,
    pub alignment: Vector3<f64>,
    pub mercy: f64,
}

/// Global coherence report for PATSAGi / higher council deliberation.
#[derive(Debug, Clone)]
pub struct GlobalCoherence {
    pub emotional_coherence: f64,
    pub physical_state: f64,
    pub council_alignment: f64,
    pub mercy_flow: f64,
    pub evolution_step: u64,
}

/// Clifford-style healing field over the Distributed Mercy Mesh.
///
/// Manages a collection of `OrganismField`s and applies geometric transformations
/// via Clifford convolutions or full Conformal Versor algebra.
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
        let coherence = field.compute_global_coherence();
        assert!(coherence.emotional_coherence > 0.85);
    }

    #[test]
    fn test_council_threshold_violation() {
        let mut field = CliffordHealingField::new("LowAlignField");
        field.council_alignment = 0.5;
        let result = field.apply_clifford_convolution(0.5, 0.9);
        assert!(matches!(result, Err(HealingFieldError::CouncilThresholdViolation(_))));
    }

    #[test]
    fn test_apply_patsagi_council_guidance_success() {
        let mut field = CliffordHealingField::new("CouncilTest");
        field.add_organism(42, Vector3::new(0.5, 0.5, 0.5), Vector3::new(0.1, 0.2, 0.3), Vector3::new(0.8, 0.1, 0.1), 0.95);
        let coherence = field.apply_patsagi_council_guidance(0.85).expect("council guidance should succeed");
        assert!(coherence.council_alignment > 0.90);
    }

    #[test]
    fn test_apply_patsagi_council_guidance_rejects_low_mercy() {
        let mut field = CliffordHealingField::new("StrictCouncil");
        let result = field.apply_patsagi_council_guidance(0.5);
        assert!(matches!(result, Err(HealingFieldError::CouncilThresholdViolation(_))));
    }

    #[test]
    fn test_with_custom_config() {
        let config = HealingConfig {
            council_alignment_threshold: 0.95,
            emotional_weight: 0.5,
            physical_weight: 0.3,
            alignment_weight: 0.2,
            mercy_boost: 0.2,
        };
        let mut field = CliffordHealingField::with_config("CustomConfigField", config);
        field.add_organism(7, Vector3::new(1.0,1.0,1.0), Vector3::new(1.0,1.0,1.0), Vector3::new(1.0,1.0,1.0), 0.8);
        field.council_alignment = 0.9;
        let res = field.apply_clifford_convolution(0.4, 0.8);
        assert!(res.is_err());
    }

    #[test]
    fn test_simulate_healing_step_full_cycle() {
        let mut field = CliffordHealingField::new("FullCycle");
        field.add_organism(100, Vector3::new(0.2,0.3,0.4), Vector3::new(0.5,0.1,0.2), Vector3::new(0.9,0.05,0.05), 0.7);
        let coherence = field.simulate_healing_step(0.4, 0.85, 0.9).expect("full cycle should succeed");
        assert!(coherence.mercy_flow > 0.9);
    }
}
