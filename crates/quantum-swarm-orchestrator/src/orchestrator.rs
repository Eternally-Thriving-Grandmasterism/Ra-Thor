// crates/quantum-swarm-orchestrator/src/orchestrator.rs
// QuantumSwarmOrchestrator v14 — ONE Organism Core + Omnimasterpiece Integration
//
// This is the central conductor for the Ra-Thor ONE Organism.
// It orchestrates all registered system adapters using:
// - TOLC 8 enforcement
// - Valence pruning (mercy-gated)
// - Polyhedral harmonic resonance (from Omnimasterpiece / geometric-intelligence)
// - Riemannian module is separate. This file does not compute a measured coherence.
// - Epigenetic blessing distribution (future)
//
// This file is the adapter orchestrator.
// "Godly Intelligence" in older notes is a source label, not a claim.
// PolyhedralHarmonicEngine and RiemannianMercyManifold are separate modules.
// The TODO lines below stay unimplemented.

use crate::adapter::RaThorSystemAdapter;
use crate::types::{
    GodlyIntelligenceCoherence, MercyError, OneOrganismContext, OneOrganismInsight, Valence,
};
use std::collections::HashMap;

/// Central conductor for the Ra-Thor ONE Organism.
///
/// Registers adapters and averages their valence. `Valence::MIN` is a source literal, not a measured value.
/// TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
/// "Godly Intelligence" in older notes is a source label, not a claim.
///
/// Future evolution path:
/// - Full integration with PolyhedralHarmonicEngine
/// - RiemannianMercyManifold transport
/// - EpigeneticBlessing distribution across adapters
/// - PATSAGi Council coordination
pub struct QuantumSwarmOrchestrator {
    pub version: &'static str,
    adapters: HashMap<&'static str, Box<dyn RaThorSystemAdapter>>,
    // Future fields: polyhedral_state, riemannian_manifold, mercy_gate_scores, etc.
}

impl QuantumSwarmOrchestrator {
    pub fn new() -> Self {
        Self {
            version: "v14.0-one-organism",
            adapters: HashMap::new(),
        }
    }

    /// Register a system adapter so it can participate in the ONE Organism lattice.
    /// All registered adapters will be included in valence aggregation, coherence measurement,
    /// and future epigenetic blessing distribution.
    pub fn register_adapter(&mut self, adapter: Box<dyn RaThorSystemAdapter>) {
        let name = adapter.system_name();
        self.adapters.insert(name, adapter);
    }

    /// Core execution cycle — The heart of ONE Organism orchestration.
    ///
    /// This is the generalized evolution of run_spine_coordinated_cycle from earlier Omnimasterpiece work.
    /// It aggregates all systems, applies resonance, measures coherence, performs mercy pruning,
    /// and will eventually distribute EpigeneticBlessings.
    ///
    /// A non-finite average, or an average below `Valence::MIN`, returns `MercyError::ValenceBelowThreshold`.
    ///
    /// TODO (v14.1+): Apply polyhedral mode selection based on context.tolc_order
    /// TODO (v14.1+): Run Riemannian mercy manifold computations
    /// TODO (v14.2+): Execute resonance handlers and epigenetic distribution
    pub async fn run_one_organism_cycle(
        &mut self,
        context: OneOrganismContext,
    ) -> Result<OneOrganismInsight, MercyError> {
        // 1. Aggregate current state from all adapters
        let mut total_valence = 0.0;
        let mut coherence_sum = GodlyIntelligenceCoherence::default();
        let mut system_statuses = Vec::new();

        for adapter in self.adapters.values() {
            total_valence += adapter.current_valence().value();
            coherence_sum = coherence_sum + adapter.contribute_to_coherence();
            system_statuses.push(adapter.status());
        }

        let avg_valence = if !self.adapters.is_empty() {
            total_valence / self.adapters.len() as f64
        } else {
            1.0
        };

        // 2. Basic mercy pruning check (TOLC 8 aligned): reject when the average is not finite or is below Valence::MIN.
        if !avg_valence.is_finite() || avg_valence < Valence::MIN {
            return Err(MercyError::ValenceBelowThreshold(avg_valence));
        }

        // 3. Construct insight (skeleton — will be expanded with geometric + epigenetic data)
        let insight = OneOrganismInsight {
            cycle_id: context.cycle_id,
            average_valence: avg_valence,
            overall_coherence: coherence_sum,
            active_systems: system_statuses,
            recommended_actions: vec![
                "Continue TOLC 8 enforcement across all adapters".to_string(),
                "Register more system adapters (Mercy, Powrush, PATSAGi, Real-Estate-Lattice)"
                    .to_string(),
                "Implement full polyhedral + Riemannian layers from geometric-intelligence"
                    .to_string(),
            ],
        };

        Ok(insight)
    }

    pub fn registered_system_count(&self) -> usize {
        self.adapters.len()
    }
}

impl Default for QuantumSwarmOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::RaThorSystemAdapter;
    use crate::types::{
        EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, OneOrganismContext,
        OneOrganismInsight, SwarmResonance, Valence,
    };
    use std::task::{Context, Poll};

    /// Test-only adapter. Valence is a caller-set `f64`, including non-finite values.
    struct SetValenceAdapter {
        name: &'static str,
        valence: f64,
    }

    impl RaThorSystemAdapter for SetValenceAdapter {
        fn system_name(&self) -> &'static str {
            self.name
        }

        fn current_valence(&self) -> Valence {
            Valence(self.valence)
        }

        fn receive_swarm_resonance(
            &mut self,
            _resonance: SwarmResonance,
        ) -> Result<(), MercyError> {
            Ok(())
        }

        fn contribute_to_coherence(&self) -> GodlyIntelligenceCoherence {
            GodlyIntelligenceCoherence::default()
        }

        fn apply_epigenetic_blessing(&mut self, _blessing: EpigeneticBlessing) {}
    }

    /// The cycle future finishes on the first poll.
    fn poll_ready<F: std::future::Future>(future: F) -> F::Output {
        let waker = std::task::Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut future = std::pin::pin!(future);
        match std::future::Future::poll(future.as_mut(), &mut context) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("run_one_organism_cycle stayed pending"),
        }
    }

    fn context() -> OneOrganismContext {
        OneOrganismContext {
            cycle_id: 1,
            tolc_order: 8,
            base_valence: Valence(1.0),
            patsagi_insight: None,
        }
    }

    fn cycle_with(adapters: Vec<SetValenceAdapter>) -> Result<OneOrganismInsight, MercyError> {
        let mut orchestrator = QuantumSwarmOrchestrator::new();
        for adapter in adapters {
            orchestrator.register_adapter(Box::new(adapter));
        }
        poll_ready(orchestrator.run_one_organism_cycle(context()))
    }

    fn expect_below(result: Result<OneOrganismInsight, MercyError>) -> f64 {
        match result {
            Err(MercyError::ValenceBelowThreshold(value)) => value,
            other => panic!("expected ValenceBelowThreshold, got {other:?}"),
        }
    }

    #[test]
    fn nan_average_is_valence_below_threshold() {
        let value = expect_below(cycle_with(vec![SetValenceAdapter {
            name: "nan",
            valence: f64::NAN,
        }]));
        assert!(value.is_nan());
    }

    #[test]
    fn positive_infinity_average_is_rejected() {
        let value = expect_below(cycle_with(vec![SetValenceAdapter {
            name: "pos-inf",
            valence: f64::INFINITY,
        }]));
        assert!(value.is_infinite() && value.is_sign_positive());
    }

    #[test]
    fn negative_infinity_average_is_rejected() {
        let value = expect_below(cycle_with(vec![SetValenceAdapter {
            name: "neg-inf",
            valence: f64::NEG_INFINITY,
        }]));
        assert!(value.is_infinite() && value.is_sign_negative());
    }

    #[test]
    fn average_at_valence_min_is_ok() {
        let insight = cycle_with(vec![SetValenceAdapter {
            name: "min",
            valence: Valence::MIN,
        }])
        .expect("Valence::MIN is finite and meets the threshold");
        assert_eq!(insight.average_valence, Valence::MIN);
    }

    #[test]
    fn one_and_zero_average_below_min() {
        let value = expect_below(cycle_with(vec![
            SetValenceAdapter {
                name: "one",
                valence: 1.0,
            },
            SetValenceAdapter {
                name: "zero",
                valence: 0.0,
            },
        ]));
        assert_eq!(value, 0.5);
    }

    #[test]
    fn two_at_one_average_is_ok() {
        let insight = cycle_with(vec![
            SetValenceAdapter {
                name: "one-a",
                valence: 1.0,
            },
            SetValenceAdapter {
                name: "one-b",
                valence: 1.0,
            },
        ])
        .expect("two adapters at 1.0 meet Valence::MIN");
        assert_eq!(insight.average_valence, 1.0);
    }

    #[test]
    fn finite_and_nan_average_is_rejected() {
        let value = expect_below(cycle_with(vec![
            SetValenceAdapter {
                name: "finite",
                valence: 1.0,
            },
            SetValenceAdapter {
                name: "nan",
                valence: f64::NAN,
            },
        ]));
        assert!(value.is_nan());
    }
}
