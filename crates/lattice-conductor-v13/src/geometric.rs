use nalgebra::{DualQuaternion, Quaternion};
use crate::{ConductorError, ConductorResult, GeometricState};

/// Geometric Motor v2 - Basic implementation with improved logic
#[derive(Debug, Default)]
pub struct BasicGeometricMotor {
    pub iterations: u32,
}

impl BasicGeometricMotor {
    pub fn new() -> Self {
        Self { iterations: 0 }
    }

    /// Internal helper to normalize a dual quaternion (for future use)
    fn normalize_dual_quaternion(&self, dq: DualQuaternion<f64>) -> DualQuaternion<f64> {
        let real_norm = dq.real().norm();
        if real_norm > 1e-9 {
            dq / real_norm
        } else {
            dq
        }
    }
}

/// Core geometric operations for the Lattice Conductor.
pub trait GeometricMotor {
    /// Apply a dual quaternion transformation.
    fn apply_dual_quaternion(&self, motor: DualQuaternion<f64>) -> ConductorResult<()>;

    /// Project using hyperbolic parameters.
    fn project_hyperbolic(&self, params: &[f64]) -> ConductorResult<()>;

    /// Enforce Study Quadric constraint on a 4D point.
    fn enforce_study_quadric(&self, point: &[f64; 4]) -> bool;
}

impl GeometricMotor for BasicGeometricMotor {
    fn apply_dual_quaternion(&self, motor: DualQuaternion<f64>) -> ConductorResult<()> {
        let real_norm = motor.real().norm();

        if real_norm < 0.01 || real_norm > 100.0 {
            return Err(ConductorError::Geometric(
                format!("Invalid dual quaternion real part norm: {:.4}", real_norm),
            ));
        }

        let dual_norm = motor.dual().norm();
        if dual_norm > 50.0 {
            return Err(ConductorError::Geometric(
                "Dual part magnitude too large for stable transformation".to_string(),
            ));
        }

        Ok(())
    }

    fn project_hyperbolic(&self, params: &[f64]) -> ConductorResult<()> {
        if params.len() != 2 {
            return Err(ConductorError::Geometric(
                "Hyperbolic projection expects exactly 2 parameters".to_string(),
            ));
        }
        Ok(())
    }

    fn enforce_study_quadric(&self, point: &[f64; 4]) -> bool {
        let norm_sq: f64 = point.iter().map(|&v| v * v).sum();
        (norm_sq - 1.0).abs() < 1e-5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_motor_rejects_invalid_dual_quaternion() {
        let motor = BasicGeometricMotor::new();

        let bad = DualQuaternion::from_real_and_dual(
            Quaternion::new(0.001, 0.0, 0.0, 0.0),
            Quaternion::identity(),
        );

        assert!(motor.apply_dual_quaternion(bad).is_err());
    }

    #[test]
    fn study_quadric_detects_unit_points() {
        let motor = BasicGeometricMotor::new();
        let unit_point = [1.0, 0.0, 0.0, 0.0];
        assert!(motor.enforce_study_quadric(&unit_point));

        let non_unit = [2.0, 0.0, 0.0, 0.0];
        assert!(!motor.enforce_study_quadric(&non_unit));
    }
}