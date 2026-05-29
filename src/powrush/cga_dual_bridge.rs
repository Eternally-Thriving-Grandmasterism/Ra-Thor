//! CGA Dual Quaternion Bridge
//!
//! Provides conversion between nalgebra's UnitDualQuaternion and
//! the CGA Motor. This allows existing dual quaternion code
//! (from geometric_self_healing and dual_rigid_healing) to interoperate
//! with the new Conformal Geometric Algebra layer.

use nalgebra::{UnitDualQuaternion, UnitQuaternion, Vector3};
use crate::powrush::cga_primitives::{Motor, Rotor, Translator};

/// Converts a nalgebra UnitDualQuaternion into a CGA Motor.
/// Note: This is an approximation for the starter implementation.
pub fn dual_quaternion_to_cga_motor(dual: &UnitDualQuaternion<f64>) -> Motor {
    let rotation = dual.rotation();
    let translation = dual.translation().vector;

    // Extract axis-angle from rotation (simplified)
    let (axis, angle) = if let Some((axis, angle)) = rotation.axis_angle() {
        (axis, angle)
    } else {
        (Vector3::z_axis().into_inner(), 0.0)
    };

    Motor {
        translator: Translator::from_vector(translation),
        rotor: Rotor::from_axis_angle(axis, angle),
    }
}

/// Converts a CGA Motor into a nalgebra UnitDualQuaternion (approximation).
pub fn cga_motor_to_dual_quaternion(motor: &Motor) -> UnitDualQuaternion<f64> {
    let rotation = UnitQuaternion::from_axis_angle(&motor.rotor.axis, motor.rotor.angle);
    let translation = motor.translator.t;

    UnitDualQuaternion::from_rotation_and_translation(rotation, translation)
}

/// Applies a dual quaternion transform to a CGA point (via conversion).
pub fn apply_dual_to_cga_point(
    dual: &UnitDualQuaternion<f64>,
    point: &crate::powrush::cga_primitives::CgaPoint,
) -> crate::powrush::cga_primitives::CgaPoint {
    let motor = dual_quaternion_to_cga_motor(dual);
    motor.apply_to_point(point)
}
