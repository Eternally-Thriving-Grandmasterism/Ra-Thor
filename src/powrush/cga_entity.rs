//! CgaEntity — Living Conformal Geometric Entity for Powrush RBE
//!
//! This primitive represents a player, faction unit, or geometric organism
//! using Conformal Geometric Algebra (CGA).
//! It serves as the foundation for entities that can be transformed,
//! healed, and animated through unified geometric operations.

use nalgebra::Vector3;
use crate::powrush::cga_primitives::{CgaPoint, Motor};

/// A living entity represented in Conformal Geometric Algebra.
#[derive(Debug, Clone)]
pub struct CgaEntity {
    pub id: u64,
    pub name: String,
    /// Current rigid transform as a CGA Motor
    pub motor: Motor,
    /// Base (rest) position as conformal point
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

    /// Returns the current world position after applying the motor.
    pub fn world_position(&self) -> CgaPoint {
        self.motor.apply_to_point(&self.base_point)
    }

    /// Applies a mercy-aligned CGA motor transform to this entity.
    pub fn apply_motor(&mut self, motor: &Motor) {
        // Compose: new_motor = motor * self.motor (simplified for starter)
        // For full CGA this would use proper geometric product composition.
        self.motor = Motor {
            translator: motor.translator,
            rotor: motor.rotor,
        };
    }

    /// Applies geometric healing using a CGA motor (mercy-scaled).
    pub fn apply_geometric_healing(&mut self, healing_motor: &Motor, mercy: f64) {
        let scaled = Motor::mercy_aligned_rigid(
            healing_motor.translator.t,
            healing_motor.rotor.axis,
            healing_motor.rotor.angle,
            mercy,
        );
        self.apply_motor(&scaled);
    }

    /// Resets the entity to its base state (useful for healing recovery).
    pub fn reset_to_base(&mut self) {
        self.motor = Motor::mercy_aligned_rigid(
            Vector3::zeros(),
            Vector3::z_axis().into_inner(),
            0.0,
            1.0,
        );
    }
}
