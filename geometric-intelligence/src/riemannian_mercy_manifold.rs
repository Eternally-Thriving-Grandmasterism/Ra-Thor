
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Z2Invariant {
    Trivial,      // 0
    NonTrivial,   // 1
}

#[derive(Debug, Clone)]
pub struct TopologicalInsulatorResponse {
    pub z2_invariant: Z2Invariant,
    pub has_protected_surface_states: bool,
    pub bulk_gap: f64,
    pub notes: String,
}

impl RiemannianMercyManifold {

    /// Analyzes topological insulator-like behavior.
    pub fn analyze_topological_insulator(
        &self,
        bulk_curvature: f64,
        surface_phase: f64,
    ) -> TopologicalInsulatorResponse {
        let bulk_gap = (bulk_curvature - 0.82).abs();

        // Simple Z2 analog: non-trivial if accumulated phase is significant and odd-like
        let z2 = if surface_phase.abs() > 0.4 {
            Z2Invariant::NonTrivial
        } else {
            Z2Invariant::Trivial
        };

        let has_protected = z2 == Z2Invariant::NonTrivial && bulk_gap > 0.1;

        let notes = match z2 {
            Z2Invariant::NonTrivial => {
                if has_protected {
                    "Non-trivial Z2 phase. Protected surface states expected (topological insulator analog).".to_string()
                } else {
                    "Non-trivial topology but bulk gap too small.".to_string()
                }
            }
            Z2Invariant::Trivial => "Trivial insulator phase. No protected surface states.".to_string(),
        };

        TopologicalInsulatorResponse {
            z2_invariant: z2,
            has_protected_surface_states: has_protected,
            bulk_gap,
            notes,
        }
    }
}
