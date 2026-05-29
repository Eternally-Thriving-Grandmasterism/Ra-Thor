//! CgaEntity — Living Conformal Geometric Entity for Powrush RBE
//!
//! Represents player organisms, faction units, or geometric beings
//! using Conformal Geometric Algebra.

use nalgebra::Vector3;
use crate::powrush::cga_primitives::{CgaPoint, Motor};

#[derive(Debug, Clone)]
pub struct CgaEntity {
    pub id: u64,
    pub name: String,
    pub motor: Motor,
    pub base_point: CgaPoint,
}

impl CgaEntity {
    pub fn new(id: u64, name: impl Into<String>, base_position: Vector3<f64>) -> Self {
        let base_point = CgaPoint::from_euclidean(
            base_position.x,
            base_position.y,
            base_position.z,
        );

        Self {
            id,
            name: name.into(),
            motor: Motor::mercy_aligned_rigid(
                Vector3::zeros(),
                Vector3::z_axis().into_inner(),
                0.0,
                1.0,
            ),
            base_point,
        }
    }

    pub fn world_position(&self) -> CgaPoint {
        self.motor.apply_to_point(&self.base_point)
    }

    /// Applies a motor transform.
    pub fn apply_motor(&mut self, motor: &Motor) {
        self.motor = self.motor.compose(motor);
    }

    /// Applies geometric healing with smooth interpolation.
    /// `progress` in [0.0, 1.0] for gradual healing effect.
    pub fn smooth_heal(&mut self, target_motor: &Motor, progress: f64, mercy: f64) {
        let healing_motor = Motor::mercy_aligned_rigid(
            target_motor.translator.t,
            target_motor.rotor.axis,
            target_motor.rotor.angle,
            mercy,
        );

        let interpolated = self.motor.slerp(&healing_motor, progress);
        self.motor = interpolated;
    }

    /// Legacy direct healing (kept for compatibility).
    pub fn apply_geometric_healing(&mut self, healing_motor: &Motor, mercy: f64) {
        self.smooth_heal(healing_motor, 1.0, mercy);
    }

    pub fn reset_to_base(&mut self) {
        self.motor = Motor::mercy_aligned_rigid(
            Vector3::zeros(),
            Vector3::z_axis().into_inner(),
            0.0,
            1.0,
        );
    }
}
