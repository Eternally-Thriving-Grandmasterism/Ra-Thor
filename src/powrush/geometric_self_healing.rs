//! src/powrush/geometric_self_healing.rs
//!
//! v14.0.9 Powrush RBE + Geometric Self-Healing (Quaternion-based)
//!
//! This module provides mercy-aligned geometric self-healing using
//! proper quaternions via nalgebra. It forms the foundation for
//! future Conformal Geometric Algebra (CGA) integration.

use nalgebra::{UnitQuaternion, Vector3};
use std::time::SystemTime;

#[derive(Debug, Clone, PartialEq)]
pub enum CliffordError {
    InvalidStrength { provided: f64 },
    InvalidAlignment { provided: f64 },
}

/// ResourceVector using nalgebra types for future extensibility.
#[derive(Debug, Clone)]
pub struct ResourceVector {
    pub scalar: f64,
    pub vector: Vector3<f64>,
    pub last_update: SystemTime,
}

impl ResourceVector {
    pub fn new(scalar: f64, vector: [f64; 3]) -> Self {
        Self {
            scalar,
            vector: Vector3::new(vector[0], vector[1], vector[2]),
            last_update: SystemTime::now(),
        }
    }

    /// Applies rotation using a proper quaternion.
    pub fn apply_rotation(&mut self, rotation: &UnitQuaternion<f64>, strength: f64) {
        self.vector = rotation * self.vector * strength;
        self.scalar *= 1.0 + (strength - 1.0) * 0.1;
        self.last_update = SystemTime::now();
    }
}

/// Mercy-aligned motor using quaternions.
#[derive(Debug, Clone)]
pub struct CliffordMotor {
    pub rotation: UnitQuaternion<f64>,
    pub scale: f64,
    pub mercy_alignment: f64,
}

impl CliffordMotor {
    pub fn mercy_aligned(strength: f64, alignment: f64) -> Result<Self, CliffordError> {
        if strength < 0.0 {
            return Err(CliffordError::InvalidStrength { provided: strength });
        }
        if !(0.0..=1.0).contains(&alignment) {
            return Err(CliffordError::InvalidAlignment { provided: alignment });
        }

        let angle = strength * alignment * 0.8;
        let axis = Vector3::new(0.3, 0.2, 0.1).normalize();

        Ok(Self {
            rotation: UnitQuaternion::from_axis_angle(&axis, angle),
            scale: 1.0 + strength * alignment,
            mercy_alignment: alignment,
        })
    }
}

/// Powrush geometric healing field with quaternion-based healing.
#[derive(Debug, Clone)]
pub struct PowrushHealingField {
    pub resource_flow_coherence: f64,
    pub faction_harmony: f64,
    pub motivation_coherence: f64,
    pub active_vectors: Vec<ResourceVector>,
}

impl PowrushHealingField {
    pub fn new() -> Self {
        Self {
            resource_flow_coherence: 0.91,
            faction_harmony: 0.87,
            motivation_coherence: 0.94,
            active_vectors: vec![ResourceVector::new(120.0, [45.0, 30.0, 15.0])],
        }
    }

    pub fn monitor_and_heal(&mut self, reason: &str) {
        let needs_healing = self.resource_flow_coherence < 0.75
            || self.faction_harmony < 0.70
            || self.motivation_coherence < 0.75;

        if needs_healing {
            if let Ok(motor) = CliffordMotor::mercy_aligned(0.35, 0.92) {
                for vector in &mut self.active_vectors {
                    vector.apply_rotation(&motor.rotation, motor.scale);
                }
            }

            self.resource_flow_coherence = (self.resource_flow_coherence + 0.12).min(1.0);
            self.faction_harmony = (self.faction_harmony + 0.09).min(1.0);
            self.motivation_coherence = (self.motivation_coherence + 0.08).min(1.0);

            println!("[Powrush] Quaternion-based geometric healing applied — {}", reason);
        }
    }
}

pub fn integrate_with_watchdog(powrush: &PowrushHealingField) {
    if powrush.motivation_coherence < 0.75 {
        println!("[Integration] Low motivation — escalating to Watchdog Level 3");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_rotation_changes_vector() {
        let mut vec = ResourceVector::new(100.0, [10.0, 20.0, 5.0]);
        let motor = CliffordMotor::mercy_aligned(0.4, 0.95).unwrap();
        let original = vec.vector;
        vec.apply_rotation(&motor.rotation, motor.scale);
        assert!(vec.vector != original);
    }

    #[test]
    fn test_mercy_aligned_error_handling() {
        assert!(CliffordMotor::mercy_aligned(-1.0, 0.8).is_err());
        assert!(CliffordMotor::mercy_aligned(0.5, 1.5).is_err());
    }
}