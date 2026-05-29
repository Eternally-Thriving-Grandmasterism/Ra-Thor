//! CgaEntity — Living Conformal Geometric Entity for Powrush RBE
//!
//! This module defines `CgaEntity`, a living object that can be
//! transformed and healed using Conformal Geometric Algebra (CGA).
//!
//! Entities are the core of future player organisms and faction units
//! in the geometric systems of Powrush.
//!
//! # Key Features
//!
//! - Position represented as `CgaPoint`
//! - Transform stored as `Motor`
//! - Smooth healing via `slerp`
//! - Convenient helpers like `translate()` and `smooth_heal()`

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

    pub fn translate(&mut self, offset: Vector3<f64>) {
        let trans_motor = Motor::mercy_aligned_rigid(offset, Vector3::z_axis().into_inner(), 0.0, 1.0);
        self.apply_motor(&trans_motor);
    }

    pub fn apply_motor(&mut self, motor: &Motor) {
        self.motor = self.motor.compose(motor);
    }

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
