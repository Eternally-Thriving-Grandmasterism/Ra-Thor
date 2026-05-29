//! Geometric Self-Healing for Powrush RBE (v14.0.9)
//!
//! Quaternion-based primitives for mercy-aligned geometric self-healing.

use std::time::SystemTime;
use nalgebra::{Vector3, UnitQuaternion};

#[derive(Debug, thiserror::Error)]
pub enum CliffordError {
    #[error("Strength must be positive")]
    InvalidStrength,
    #[error("Alignment must be between 0.0 and 1.0")]
    InvalidAlignment,
    #[error("Invalid vector operation")]
    InvalidOperation,
}

#[derive(Debug, Clone)]
pub struct ResourceVector {
    pub scalar: f64,
    pub vector: Vector3<f64>,
    pub last_update: SystemTime,
}

impl ResourceVector {
    pub fn new(scalar: f64, vector: Vector3<f64>) -> Self {
        Self { scalar, vector, last_update: SystemTime::now() }
    }

    pub fn apply_rotation(&mut self, rotation: &UnitQuaternion<f64>, strength: f64) -> Result<(), CliffordError> {
        if strength <= 0.0 { return Err(CliffordError::InvalidStrength); }
        self.vector = rotation * self.vector;
        self.scalar *= 1.0 + (strength - 1.0) * 0.1;
        self.last_update = SystemTime::now();
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CliffordMotor {
    pub rotation: UnitQuaternion<f64>,
    pub scale: f64,
    pub mercy_alignment: f64,
}

impl CliffordMotor {
    pub fn mercy_aligned(strength: f64, alignment: f64) -> Result<Self, CliffordError> {
        if strength <= 0.0 { return Err(CliffordError::InvalidStrength); }
        if !(0.0..=1.0).contains(&alignment) { return Err(CliffordError::InvalidAlignment); }

        let axis = Vector3::new(0.3, 0.2, 0.1).normalize();
        let angle = strength * alignment * std::f64::consts::FRAC_PI_2;
        let rotation = UnitQuaternion::from_axis_angle(&axis, angle);

        Ok(Self { rotation, scale: strength, mercy_alignment: alignment })
    }
}

#[derive(Debug)]
pub struct PowrushHealingField {
    pub resource_flow_coherence: f64,
    pub faction_harmony: f64,
    pub motivation_coherence: f64,
    pub active_vectors: Vec<ResourceVector>,
}

impl PowrushHealingField {
    pub fn new() -> Self {
        Self {
            resource_flow_coherence: 0.85,
            faction_harmony: 0.80,
            motivation_coherence: 0.75,
            active_vectors: vec![ResourceVector::new(1.0, Vector3::new(1.0, 0.0, 0.0))],
        }
    }

    pub fn monitor_and_heal(&mut self, reason: &str) -> Result<(), CliffordError> {
        let needs_healing = self.resource_flow_coherence < 0.70
            || self.faction_harmony < 0.70
            || self.motivation_coherence < 0.70;

        if needs_healing {
            let motor = CliffordMotor::mercy_aligned(1.1, 0.9)?;
            for vector in &mut self.active_vectors {
                vector.apply_rotation(&motor.rotation, motor.scale)?;
            }
            self.resource_flow_coherence = (self.resource_flow_coherence + 0.15).min(1.0);
            self.faction_harmony = (self.faction_harmony + 0.12).min(1.0);
            self.motivation_coherence = (self.motivation_coherence + 0.18).min(1.0);
            println!("[PowrushHealingField] Geometric healing applied. Reason: {}", reason);
        }
        Ok(())
    }

    pub fn integrate_with_watchdog(&self) {
        if self.motivation_coherence < 0.75 {
            println!("Watchdog: Low motivation coherence detected.");
        }
    }
}
