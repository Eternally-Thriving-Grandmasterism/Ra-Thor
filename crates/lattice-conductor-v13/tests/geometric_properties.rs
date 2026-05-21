use lattice_conductor_v13::geometric::{BasicGeometricMotor, GeometricMotor};
use nalgebra::DualQuaternion;
use proptest::prelude::*;

// Simple strategy for dual quaternions (basic version for now)
fn arb_dual_quaternion() -> impl Strategy<Value = DualQuaternion<f64>> {
    prop::array::uniform8(-2.0f64..2.0).prop_map(|arr| {
        DualQuaternion::from_real_and_dual(
            nalgebra::Quaternion::new(arr[0], arr[1], arr[2], arr[3]),
            nalgebra::Quaternion::new(arr[4], arr[5], arr[6], arr[7]),
        )
    })
}

proptest! {
    #[test]
    fn study_quadric_enforcement_works(point in prop::array::uniform4(-5.0f64..5.0)) {
        let motor = BasicGeometricMotor::new();
        let result = motor.enforce_study_quadric(&point);
        // For normalized points it should return true
        if (point.iter().map(|x| x*x).sum::<f64>() - 1.0).abs() < 1e-4 {
            prop_assert!(result);
        }
    }

    #[test]
    fn apply_dual_quaternion_does_not_crash(motor in arb_dual_quaternion()) {
        let geo_motor = BasicGeometricMotor::new();
        let _ = geo_motor.apply_dual_quaternion(motor);
        // Should not panic
    }
}