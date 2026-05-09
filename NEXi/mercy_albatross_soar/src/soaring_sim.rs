// mercy_albatross_soar/src/soaring_sim.rs — Numerical Dynamic Soaring Simulation
use std::f64::consts::PI;

#[derive(Clone, Copy)]
struct State {
    h: f64,      // height (m)
    x: f64,      // horizontal position (m)
    v_a: f64,    // airspeed (m/s)
    gamma: f64,  // flight path angle (rad)
    psi: f64,    // heading relative to wind (rad)
}

pub fn simulate_dynamic_soaring(cycles: u32) -> f64 {
    let mut state = State {
        h: 10.0,
        x: 0.0,
        v_a: 20.0,
        gamma: 0.0,
        psi: PI / 2.0, // start crosswind
    };

    let dt = 0.1; // time step (s)
    let g = 9.81;
    let rho = 1.225;
    let s = 0.8;
    let m = 10.0;
    let c_d0 = 0.015;
    let k = 0.04;
    let v_ref = 5.0;
    let h_ref = 10.0;
    let z0 = 0.001;

    let mut energy_start = 0.5 * m * state.v_a.powi(2) + m * g * state.h;
    let mut total_gain = 0.0;

    for cycle in 0..cycles {
        let mut t = 0.0;
        let cycle_duration = 20.0; // approximate cycle time (s)

        while t < cycle_duration {
            // Wind speed at height
            let v_w = v_ref * (state.h + z0).ln() / (h_ref + z0).ln();

            // Lift & drag (simple model)
            let c_l = 1.2; // cruise C_L
            let c_d = c_d0 + k * c_l.powi(2);
            let lift = 0.5 * rho * state.v_a.powi(2) * s * c_l;
            let drag = 0.5 * rho * state.v_a.powi(2) * s * c_d;

            // Bank angle control (simple sinusoidal for demo)
            let phi = 0.8 * (t * 2.0 * PI / cycle_duration).sin(); // ±45° approx

            // Equations of motion (simplified)
            let dv_dt = g * (state.gamma.sin() - (c_d / c_l) * state.gamma.cos())
                + (v_ref / (h_ref + z0)) * state.v_a * state.gamma.sin() * state.psi.cos();
            let dgamma_dt = (g / state.v_a) * (c_l * phi.cos() - state.gamma.cos());
            let dpsi_dt = (g / state.v_a) * (c_l * phi.sin() / state.gamma.cos());

            // Update state (Euler step for simplicity)
            state.v_a += dv_dt * dt;
            state.gamma += dgamma_dt * dt;
            state.psi += dpsi_dt * dt;
            state.h += state.v_a * state.gamma.sin() * dt;
            state.x += (state.v_a * state.gamma.cos() * state.psi.cos() + v_w) * dt;

            t += dt;
        }

        let energy_end = 0.5 * m * state.v_a.powi(2) + m * g * state.h;
        let gain = energy_end - energy_start;
        total_gain += gain;
        energy_start = energy_end;

        println!("Cycle {}: net energy gain = {:.2} J", cycle + 1, gain);
    }

    total_gain / cycles as f64
}

pub fn run_soaring_sim() {
    let avg_gain = simulate_dynamic_soaring(10);
    println!("Average energy gain per cycle: {:.2} J", avg_gain);
}
