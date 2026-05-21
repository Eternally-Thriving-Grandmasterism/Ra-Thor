use lattice_conductor_v13::geometric::{BasicGeometricMotor, GeometricMotor};
use nalgebra::DualQuaternion;
use proptest::prelude::*;

// --- Strategy helpers ---
fn arb_dual_quaternion() -> impl Strategy<Value = DualQuaternion<f64>> {
    prop::array::uniform8(-2.0f64..2.0).prop_map(|arr| {
        DualQuaternion::from_real_and_dual(
            nalgebra::Quaternion::new(arr[0], arr[1], arr[2], arr[3]),
            nalgebra::Quaternion::new(arr[4], arr[5], arr[6], arr[7]),
        )
    })
}

fn arb_unit_like_dual_quaternion() -> impl Strategy<Value = DualQuaternion<f64>> {
    // Generates dual quaternions with reasonable real part norm
    prop::array::uniform8(-1.5f64..1.5).prop_map(|arr| {
        let mut real = nalgebra::Quaternion::new(arr[0], arr[1], arr[2], arr[3]);
        if real.norm() < 0.1 { real = nalgebra::Quaternion::identity(); }
        DualQuaternion::from_real_and_dual(real, nalgebra::Quaternion::new(arr[4], arr[5], arr[6], arr[7]))
    })
}

proptest! {
    #[test]
    fn study_quadric_enforcement_works(point in prop::array::uniform4(-5.0f64..5.0)) {
        let motor = BasicGeometricMotor::new();
        let result = motor.enforce_study_quadric(&point);
        if (point.iter().map(|x| x*x).sum::<f64>() - 1.0).abs() < 1e-4 {
            prop_assert!(result);
        }
    }

    #[test]
    fn apply_dual_quaternion_accepts_reasonable_inputs(motor in arb_unit_like_dual_quaternion()) {
        let geo_motor = BasicGeometricMotor::new();
        let result = geo_motor.apply_dual_quaternion(motor);
        prop_assert!(result.is_ok());
    }

    #[test]
    fn apply_dual_quaternion_rejects_bad_magnitude() {
        let geo_motor = BasicGeometricMotor::new();

        // Very small real part
        let bad_small = DualQuaternion::from_real_and_dual(
            nalgebra::Quaternion::new(0.001, 0.0, 0.0, 0.0),
            nalgebra::Quaternion::identity(),
        );
        prop_assert!(geo_motor.apply_dual_quaternion(bad_small).is_err());

        // Very large real part
        let bad_large = DualQuaternion::from_real_and_dual(
            nalgebra::Quaternion::new(500.0, 0.0, 0.0, 0.0),
            nalgebra::Quaternion::identity(),
        );
        prop_assert!(geo_motor.apply_dual_quaternion(bad_large).is_err());
    }

    #[test]
    fn hyperbolic_projection_requires_correct_params() {
        let motor = BasicGeometricMotor::new();
        prop_assert!(motor.project_hyperbolic(&[1.0, 2.0]).is_ok());
        prop_assert!(motor.project_hyperbolic(&[1.0]).is_err());
    }
}