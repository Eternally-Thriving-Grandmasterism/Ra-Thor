// mercy_numerical/src/lie_group_integrators.rs — Mercy-Gated Lie-Group Integrators
use nalgebra::{Matrix3, Vector3, Isometry3};
use nalgebra::geometry::{Rotation3, Translation3};

type SO3 = Rotation3<f64>;
type SE3 = Isometry3<f64>;

/// Runge-Kutta-Munthe-Kaas order 4 for Lie groups (explicit)
pub fn rkmk4<F>(f: F, t0: f64, y0: SE3, tf: f64, h: f64) -> Vec<(f64, SE3)>
where
    F: Fn(f64, &SE3) -> Vector3<f64>, // Lie algebra element (se(3) twist)
{
    let mut t = t0;
    let mut y = y0;
    let mut result = vec![(t, y)];

    let c = [0.0, 0.5, 0.5, 1.0];
    let b = [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0];
    let a = [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];

    while t < tf {
        let mut k = [Vector3::zeros(); 4];

        for i in 0..4 {
            let mut sum = Vector3::zeros();
            for j in 0..i {
                sum += a[i][j] * k[j];
            }
            let y_temp = y * SE3::from_parts(Translation3::from(sum), SO3::identity());
            k[i] = f(t + h * c[i], &y_temp);
        }

        let mut sum = Vector3::zeros();
        for i in 0..4 {
            sum += b[i] * k[i];
        }

        // Exponential map approximation for SE(3)
        let twist = sum;
        let rot = SO3::from_axis_angle(&twist.fixed_slice::<3,1>(3,0), twist.norm());
        let trans = Translation3::from(twist.fixed_slice::<3,1>(0,0));
        y = y * SE3::from_parts(trans, rot);

        t += h;
        result.push((t, y));
    }

    result
}
