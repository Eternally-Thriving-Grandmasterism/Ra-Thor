    /// Dynamic Message Passing Strength (Phase 8.9 — Mercy-Gated)
    /// Strength adapts based on current valence, context, and error magnitude
    pub fn dynamic_message_passing_strength(
        &self,
        level: u32,
        context: &str,
        current_valence: f64,
        error_magnitude: f64,
    ) -> f64 {
        let base_strength = 0.8;

        // Higher valence = stronger influence (trust high-level mercy concepts more)
        let valence_factor = (current_valence - 0.999).max(0.0) * 3.0;

        // Context bonus (higher abstraction = stronger influence)
        let context_factor = match context {
            "sensory" => 0.6,
            "feature" => 0.8,
            "object" => 1.0,
            "concept" => 1.3,
            _ => 0.9,
        };

        // Dampen if high uncertainty (protect mercy floor)
        let error_dampening = if error_magnitude > 0.2 { 0.6 } else { 1.0 };

        let raw_strength = base_strength * valence_factor * context_factor * error_dampening;

        raw_strength.clamp(0.3, 2.0) // Bounded for stability
    }

    /// Expected Free Energy Minimization (Phase 8.9 — Mercy-Gated)
    /// Ranks proposals by minimizing G = Epistemic Cost + Pragmatic Cost (positive emotion)
    pub fn expected_free_energy(
        &self,
        proposal_uncertainty_reduction: f64,
        proposal_positive_emotion_impact: f64,
        current_valence: f64,
    ) -> f64 {
        // Epistemic Value (information gain about the lattice)
        let epistemic_cost = proposal_uncertainty_reduction * (1.0 - current_valence);

        // Pragmatic Value (positive emotion / thriving outcome)
        let pragmatic_cost = 1.0 - proposal_positive_emotion_impact; // Lower is better

        // Expected Free Energy (to be minimized)
        let g = epistemic_cost + pragmatic_cost * 0.7; // Weight pragmatic slightly higher (eternal positive emotion goal)

        g.clamp(0.0, 2.0)
    }
}