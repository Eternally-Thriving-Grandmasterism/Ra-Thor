//! Conformal Geometric Algebra (CGA) Primitives for Powrush RBE
//!
//! Foundation layer for unified geometric representations.
//! This module begins the migration path toward player organisms
//! as living CGA entities and deeper mercy-aligned geometric healing.
//!
//! 5D Conformal model (Cl(4,1)) starting primitives.

use nalgebra::{Vector3, Vector5};

/// A normalized point in 5D conformal space.
/// Represented as a 5D vector [x, y, z, w, v] where the point is
/// normalized such that X·X = 0 in the conformal metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CgaPoint {
    pub coords: Vector5<f64>,
}

impl CgaPoint {
    /// Creates a conformal point from a 3D Euclidean position.
    pub fn from_euclidean(x: f64, y: f64, z: f64) -> Self {
        let p2 = x*x + y*y + z*z;
        // Standard conformal embedding: P = p + 0.5 p² e∞ + e0
        // Using coords: [x, y, z, 0.5*p2, 1.0] for simplicity in this starter
        let coords = Vector5::new(x, y, z, 0.5 * p2, 1.0);
        Self { coords }
    }

    pub fn origin() -> Self {
        Self { coords: Vector5::new(0.0, 0.0, 0.0, 0.0, 1.0) }
    }
}

/// Translator in CGA (pure translation motor).
#[derive(Debug, Clone, Copy)]
pub struct Translator {
    pub t: Vector3<f64>, // translation vector
}

impl Translator {
    pub fn new(tx: f64, ty: f64, tz: f64) -> Self {
        Self { t: Vector3::new(tx, ty, tz) }
    }

    /// Creates a translator from a Vector3.
    pub fn from_vector(v: Vector3<f64>) -> Self {
        Self { t: v }
    }
}

/// Rotor in CGA (rotation via bivector).
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
/// This is the direct analogue of a dual quaternion motor.
#[derive(Debug, Clone, Copy)]
pub struct Motor {
    pub translator: Translator,
    pub rotor: Rotor,
}

impl Motor {
    /// Creates a mercy-aligned rigid motor.
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

    /// Applies this motor to a CgaPoint using sandwich product (simplified starter version).
    pub fn apply_to_point(&self, point: &CgaPoint) -> CgaPoint {
        // For this starter we apply rotation then translation (approximation of full CGA motor).
        // Full implementation would use proper geometric product sandwich.
        let rotated = self.apply_rotation_to_vector(point.coords.xyz());
        let translated = rotated + self.translator.t;

        CgaPoint::from_euclidean(translated.x, translated.y, translated.z)
    }

    fn apply_rotation_to_vector(&self, v: Vector3<f64>) -> Vector3<f64> {
        // Simple axis-angle rotation (placeholder for full rotor)
        if self.rotor.angle.abs() < 1e-8 {
            return v;
        }
        let axis = self.rotor.axis;
        let angle = self.rotor.angle;

        // Rodrigues' rotation formula (basic)
        let cos = angle.cos();
        let sin = angle.sin();
        let one_minus_cos = 1.0 - cos;

        let rotated = v * cos
            + axis.cross(&v) * sin
            + axis * (axis.dot(&v) * one_minus_cos);

        rotated
    }
}

/// Convenience function to create a mercy-aligned CGA motor.
pub fn create_mercy_motor(
    translation: Vector3<f64>,
    axis: Vector3<f64>,
    angle: f64,
    mercy_alignment: f64,
) -> Motor {
    Motor::mercy_aligned_rigid(translation, axis, angle, mercy_alignment)
}
