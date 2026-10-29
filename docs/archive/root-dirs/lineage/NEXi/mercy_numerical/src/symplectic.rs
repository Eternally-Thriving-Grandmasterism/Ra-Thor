// mercy_numerical/src/symplectic.rs — Mercy-Gated Symplectic Integrators
use nalgebra::{VectorN, DimName, DimMin, U6};

type State6 = VectorN<f64, U6>;

/// Velocity Verlet (2nd order symplectic, explicit, for separable Hamiltonians)
pub fn velocity_verlet<F, G>(f: F, g: G, t0: f64, y0: State6, tf: f64, h: f64) -> Vec<(f64, State6)>
where
    F: Fn(f64, &State6) -> State6, // force = f(t, y) → acceleration
    G: Fn(f64, &State6) -> State6, // not needed if force independent of velocity
{
    let mut t = t0;
    let mut y = y0;
    let mut v = y[3..6].into(); // assume y = [q, p] or [x, v]
    let mut result = vec![(t, y)];

    while t < tf {
        let a = f(t, &y);
        let v_half = v + 0.5 * h * a;
        let q_new = y[0..3] + h * v_half;
        let a_new = f(t + h, &q_new.into());
        let v_new = v_half + 0.5 * h * a_new;

        y[0..3] = q_new;
        v = v_new;
        y[3..6] = v_new.into();

        t += h;
        result.push((t, y));
    }
    result
}

/// Yoshida 4th order composition of Velocity Verlet
pub fn yoshida_fr4<F>(f: F, t0: f64, y0: State6, tf: f64, h: f64) -> Vec<(f64, State6)>
where
    F: Fn(f64, &State6) -> State6,
{
    let w0 = -2f64.powf(1.0 / 3.0) / (2.0 - 2f64.powf(1.0 / 3.0));
    let w1 = 1.0 / (2.0 - 2f64.powf(1.0 / 3.0));

    let weights = [w0, w1, w1, w0];

    let mut t = t0;
    let mut y = y0;
    let mut result = vec![(t, y)];

    for &w in &weights {
        let h_step = w * h;
        let mut temp = y.clone();
        let v_half = y[3..6] + 0.5 * h_step * f(t, &y);
        temp[0..3] += h_step * v_half;
        let a_new = f(t + h_step, &temp);
        temp[3..6] = v_half + 0.5 * h_step * a_new;

        y = temp;
        t += h_step;
        result.push((t, y));
    }

    result
}
