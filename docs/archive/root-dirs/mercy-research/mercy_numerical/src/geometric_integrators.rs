// mercy_numerical/src/geometric_integrators.rs — Mercy-Gated Geometric Integrators
use nalgebra::{VectorN, DimName, U6};

type State6 = VectorN<f64, U6>;

/// Velocity Verlet — 2nd order symplectic, explicit, separable H = T(p) + V(q)
pub fn velocity_verlet<F>(f: F, t0: f64, y0: State6, tf: f64, h: f64) -> Vec<(f64, State6)>
where
    F: Fn(f64, &State6) -> State6, // acceleration = f(t, position)
{
    let mut t = t0;
    let mut y = y0;
    let mut v = y.fixed_slice::<3,1>(3,0).into_owned(); // assume y = [q, v]
    let mut result = vec![(t, y)];

    while t < tf {
        let a = f(t, &y);
        let v_half = v + 0.5 * h * a.fixed_slice::<3,1>(0,0);
        let q_new = y.fixed_slice::<3,1>(0,0) + h * v_half;
        let a_new = f(t + h, &q_new.insert_rows(3, v_half));
        let v_new = v_half + 0.5 * h * a_new.fixed_slice::<3,1>(0,0);

        y.fixed_rows_mut::<3>(0).copy_from(&q_new);
        v = v_new;
        y.fixed_rows_mut::<3>(3).copy_from(&v);

        t += h;
        result.push((t, y));
    }
    result
}

/// Yoshida 4th-order composition of Velocity Verlet (Forest-Ruth)
pub fn yoshida_fr4<F>(f: F, t0: f64, y0: State6, tf: f64, h: f64) -> Vec<(f64, State6)>
where
    F: Fn(f64, &State6) -> State6,
{
    let w0 = -2.0_f64.powf(1.0 / 3.0) / (2.0 - 2.0_f64.powf(1.0 / 3.0));
    let w1 = 1.0 / (2.0 - 2.0_f64.powf(1.0 / 3.0));
    let weights = [w0, w1, w1, w0];

    let mut t = t0;
    let mut y = y0;
    let mut result = vec![(t, y)];

    for &w in &weights {
        let h_step = w * h;
        let mut temp = y;
        let v_half = y.fixed_slice::<3,1>(3,0) + 0.5 * h_step * f(t, &y).fixed_slice::<3,1>(0,0);
        temp.fixed_slice_mut::<3,1>(0,0) += h_step * v_half;
        let a_new = f(t + h_step, &temp);
        temp.fixed_slice_mut::<3,1>(3,0) = v_half + 0.5 * h_step * a_new.fixed_slice::<3,1>(0,0);

        y = temp;
        t += h_step;
        result.push((t, y));
    }
    result
}
