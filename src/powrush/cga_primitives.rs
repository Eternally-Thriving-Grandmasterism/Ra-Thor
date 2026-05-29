//! Conformal Geometric Algebra (CGA) Primitives for Powrush RBE
//!
//! Foundation layer for unified geometric representations.
//! This module begins the migration path toward player organisms
//! as living CGA entities and deeper mercy-aligned geometric healing.
//!
//! 5D Conformal model (Cl(4,1)) starting primitives.

use nalgebra::{Vector3, Vector5};

/// A normalized point in 5D conformal space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CgaPoint {
    pub coords: Vector5<f64>,
}

impl CgaPoint {
    pub fn from_euclidean(x: f64, y: f64, z: f64) -> Self {
        let p2 = x*x + y*y + z*z;
        let coords = Vector5::new(x, y, z, 0.5 * p2, 1.0);
        Self { coords }
    }

    pub fn origin() -> Self {
        Self { coords: Vector5::new(0.0, 0.0, 0.0, 0.0, 1.0) }
    }
}

/// Translator in CGA.
#[derive(Debug, Clone, Copy)]
pub struct Translator {
    pub t: Vector3<f64>,
}

impl Translator {
    pub fn new(tx: f64, ty: f64, tz: f64) -> Self {
        Self { t: Vector3::new(tx, ty, tz) }
    }

    pub fn from_vector(v: Vector3<f64>) -> Self {
        Self { t: v }
    }
}

/// Rotor in CGA.
#[derive(Debug, Clone, Copy)]
pub struct Rotor {
    pub angle: f64,
    pub axis: Vector3<f64>,
}

impl Rotor {
    pub fn from_axis_angle(axis: Vector3<f64>, angle: f64) -> Self {
        Self {
            axis: axis.normalize(),
            angle,
        }
    }
}

/// Motor = combined Translator + Rotor (CGA rigid transform).
/// Now supports composition and smooth interpolation.
#[derive(Debug, Clone, Copy)]
pub struct Motor {
    pub translator: Translator,
    pub rotor: Rotor,
}

impl Motor {
    pub fn mercy_aligned_rigid(
        translation: Vector3<f64>,
        axis: Vector3<f64>,
        angle: f64,
        mercy: f64,
    ) -> Self {
        let scaled_trans = translation * mercy;
        let scaled_angle = angle * mercy;

        Self {
            translator: Translator::from_vector(scaled_trans),
            rotor: Rotor::from_axis_angle(axis, scaled_angle),
        }
    }

    /// Composes two motors: self after other (other applied first).
    pub fn compose(&self, other: &Motor) -> Motor {
        // Simplified composition: combine translations and rotations
        let new_translation = self.translator.t + other.translator.t;
        let new_angle = self.rotor.angle + other.rotor.angle;
        let new_axis = if self.rotor.angle.abs() > other.rotor.angle.abs() {
            self.rotor.axis
        } else {
            other.rotor.axis
        };

        Motor {
            translator: Translator::from_vector(new_translation),
            rotor: Rotor::from_axis_angle(new_axis, new_angle),
        }
    }

    /// Spherical linear interpolation between two motors.
    /// Useful for smooth animation and healing transitions.
    pub fn slerp(&self, other: &Motor, t: f64) -> Motor {
        let t = t.clamp(0.0, 1.0);

        // Interpolate translation linearly
        let interp_translation = self.translator.t * (1.0 - t) + other.translator.t * t;

        // Interpolate rotation using simple slerp approximation
        let interp_angle = self.rotor.angle * (1.0 - t) + other.rotor.angle * t;
        let interp_axis = if self.rotor.axis.dot(&other.rotor.axis) > 0.0 {
            self.rotor.axis.lerp(&other.rotor.axis, t).normalize()
        } else {
            self.rotor.axis
        };

        Motor {
            translator: Translator::from_vector(interp_translation),
            rotor: Rotor::from_axis_angle(interp_axis, interp_angle),
        }
    }

    pub fn apply_to_point(&self, point: &CgaPoint) -> CgaPoint {
        let rotated = self.apply_rotation_to_vector(point.coords.xyz());
        let translated = rotated + self.translator.t;
        CgaPoint::from_euclidean(translated.x, translated.y, translated.z)
    }

    fn apply_rotation_to_vector(&self, v: Vector3<f64>) -> Vector3<f64> {
        if self.rotor.angle.abs() < 1e-8 {
            return v;
        }
        let axis = self.rotor.axis;
        let angle = self.rotor.angle;

        let cos = angle.cos();
        let sin = angle.sin();
        let one_minus_cos = 1.0 - cos;

        v * cos
            + axis.cross(&v) * sin
            + axis * (axis.dot(&v) * one_minus_cos)
    }
}

/// Convenience function.
pub fn create_mercy_motor(
    translation: Vector3<f64>,
    axis: Vector3<f64>,
    angle: f64,
    mercy_alignment: f64,
) -> Motor {
    Motor::mercy_aligned_rigid(translation, axis, angle, mercy_alignment)
}
