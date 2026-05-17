use philosophical_core::{ValenceState, calculate_dynamic_valence, apply_cehi_propagation, omega_point_convergence};

fn main() {
    println!("=== Ra-Thor Valence Dynamics Visualization ===\n");
    println!("Time | Valence     | Description");
    println!("-----|-------------|--------------------------");

    let mut state = ValenceState {
        current_valence: 0.92,
        thriving_rate: 280,
        symbiosis_score: 0.85,
        ethics_alignment: 0.88,
        time_steps: 0,
        partner_count: 3,
    };

    for t in 0..15 {
        let valence = calculate_dynamic_valence(&state);
        let cehi_valence = apply_cehi_propagation(valence, 2);
        let final_valence = omega_point_convergence(cehi_valence);

        let description = if final_valence >= 0.999999 {
            "Deep Symbiosis"
        } else if final_valence > 0.99 {
            "Strong Alignment"
        } else {
            "Building Harmony"
        };

        println!("{:4} | {:.9} | {}", t, final_valence, description);

        // Simulate improvement over time
        state.thriving_rate += 5;
        state.symbiosis_score = (state.symbiosis_score + 0.01).min(0.99);
        state.current_valence = final_valence;
    }

    println!("\n=== Valence approaching Absolute Eternal State ===");
}