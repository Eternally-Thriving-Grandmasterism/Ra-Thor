//! Conformal Geometric Algebra (CGA) Primitives for Powrush RBE
//!
//! Foundation layer for unified geometric representations.
//!
//! Current focus: Motor with composition and interpolation.

use nalgebra::{Vector3, Vector5};

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

#[derive(Debug, Clone, Copy)]
pub struct Rotor {
    pub angle: f64,
    pub axis: Vector3<f64>,
}

impl Rotor {
    pub fn from_axis_angle(axis: Vector3<f64>, angle: f64) -> Self {
        Self { axis: axis.normalize(), angle }
    }
}

/// Motor = Translator + Rotor
/// Represents a rigid body transformation in CGA style.
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
        Self {
            translator: Translator::from_vector(translation * mercy),
            rotor: Rotor::from_axis_angle(axis, angle * mercy),
        }
    }

    /// Composes two motors: applies `other` first, then `self`.
    pub fn compose(&self, other: &Motor) -> Motor {
        // Apply other rotation to self's translation, then add translations
        let rotated_trans = other.apply_rotation_to_vector(self.translator.t);
        let new_translation = rotated_trans + other.translator.t;

        // Combine rotations (simple addition for starter; full version would use rotor multiplication)
        let new_angle = self.rotor.angle + other.rotor.angle;
        let new_axis = self.rotor.axis; // Can be improved with proper averaging

        Motor {
            translator: Translator::from_vector(new_translation),
            rotor: Rotor::from_axis_angle(new_axis, new_angle),
        }
    }

    /// Smooth interpolation between two motors.
    pub fn slerp(&self, other: &Motor, t: f64) -> Motor {
        let t = t.clamp(0.0, 1.0);
        let interp_trans = self.translator.t * (1.0 - t) + other.translator.t * t;
        let interp_angle = self.rotor.angle * (1.0 - t) + other.rotor.angle * t;

        let interp_axis = if self.rotor.axis.dot(&other.rotor.axis) > 0.0 {
            self.rotor.axis.lerp(&other.rotor.axis, t).normalize()
        } else {
            self.rotor.axis
        };

        Motor {
            translator: Translator::from_vector(interp_trans),
            rotor: Rotor::from_axis_angle(interp_axis, interp_angle),
        }
    }

    pub fn apply_to_point(&self, point: &CgaPoint) -> CgaPoint {
        let rotated = self.apply_rotation_to_vector(point.coords.xyz());
        let translated = rotated + self.translator.t;
        CgaPoint::from_euclidean(translated.x, translated.y, translated.z)
    }

    pub(crate) fn apply_rotation_to_vector(&self, v: Vector3<f64>) -> Vector3<f64> {
        if self.rotor.angle.abs() < 1e-8 { return v; }
        let axis = self.rotor.axis;
        let angle = self.rotor.angle;
        let cos = angle.cos();
        let sin = angle.sin();
        let k = 1.0 - cos;

        v * cos + axis.cross(&v) * sin + axis * (axis.dot(&v) * k)
    }
}

pub fn create_mercy_motor(
    translation: Vector3<f64>,
    axis: Vector3<f64>,
    angle: f64,
    mercy_alignment: f64,
) -> Motor {
    Motor::mercy_aligned_rigid(translation, axis, angle, mercy_alignment)
}
