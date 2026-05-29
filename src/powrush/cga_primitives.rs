//! Conformal Geometric Algebra Primitives for Powrush RBE
//!
//! Foundational CGA types (Motor, CgaPoint, CgaSphere) with mercy-aligned construction.

use nalgebra::{Vector3, UnitQuaternion};
use std::f64::consts::FRAC_PI_2;

#[derive(Debug, Clone)]
pub struct CgaPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl CgaPoint {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn to_euclidean(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone)]
pub struct Motor {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
    pub mercy_alignment: f64,
}

impl Motor {
    pub fn mercy_aligned_rigid(
        direction: Vector3<f64>,
        up: Vector3<f64>,
        strength: f64,
        mercy: f64,
    ) -> Self {
        let axis = direction.normalize();
        let angle = strength * mercy * FRAC_PI_2;
        let rotation = UnitQuaternion::from_axis_angle(&axis, angle);

        Self {
            rotation,
            translation: direction * strength * 0.5,
            mercy_alignment: mercy,
        }
    }

    pub fn compose(&self, other: &Motor) -> Motor {
        let new_rotation = self.rotation * other.rotation;
        let rotated_trans = self.rotation * other.translation;
        let new_translation = self.translation + rotated_trans;

        Motor {
            rotation: new_rotation,
            translation: new_translation,
            mercy_alignment: (self.mercy_alignment + other.mercy_alignment) * 0.5,
        }
    }

    pub fn apply_to_point(&self, point: &CgaPoint) -> CgaPoint {
        let v = Vector3::new(point.x, point.y, point.z);
        let rotated = self.rotation * v;
        let translated = rotated + self.translation;
        CgaPoint::new(translated.x, translated.y, translated.z)
    }

    pub fn slerp(&self, other: &Motor, t: f64) -> Motor {
        let rot = self.rotation.slerp(&other.rotation, t);
        let trans = self.translation.lerp(&other.translation, t);
        Motor {
            rotation: rot,
            translation: trans,
            mercy_alignment: self.mercy_alignment * (1.0 - t) + other.mercy_alignment * t,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CgaSphere {
    pub center: CgaPoint,
    pub radius: f64,
}

impl CgaSphere {
    pub fn new(center: CgaPoint, radius: f64) -> Self {
        Self { center, radius }
    }

    pub fn apply_motor(&mut self, motor: &Motor) {
        self.center = motor.apply_to_point(&self.center);
    }
}
