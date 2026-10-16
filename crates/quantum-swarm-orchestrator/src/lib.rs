//! `ra-thor-quantum-swarm-orchestrator` compiles the adapter orchestrator and the local geometric modules.
//! `Valence::MIN` is the source literal `0.9999999`, not a measured value.
//! Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
//! `SEVEN_LIVING_MERCY_GATES` uses the seven names in `tolc_seven_mercy_gates.rs`. Those names are not that list.
//! Polyhedral mode, Riemannian manifold transport, and resonance handlers in `orchestrator.rs` stay TODO.
//! `self-evolution` is not a dependency. `../../self-evolution` has no Cargo.toml.
//! Files that are not modules carry the STUB-AUDIT-1 QSO-BUILDS-1 design-only header.
//! Contact: info@Rathor.ai

pub mod adapter;
pub mod adapters;
pub mod fractal_topology_engine;
pub mod fractal_valence_commitment;
pub mod orchestrator;
pub mod polyhedral_harmonic_engine;
pub mod riemannian_manifold;
pub mod riemannian_mercy_manifold;
pub mod tolc_seven_mercy_gates;
pub mod types;

pub use adapter::RaThorSystemAdapter;
pub use adapters::{
    CryptoSystemAdapter, FractalMercyLedgerAdapter, GeometricResonanceReport,
    LatticeConductorAdapter,
};
pub use fractal_topology_engine::{FractalShard, FractalTopologyEngine, FractalTopologyReport};
pub use fractal_valence_commitment::{
    build_valence_commitment_tree, verify_valence_commitment, ValenceCommitmentNode,
};
pub use orchestrator::QuantumSwarmOrchestrator;
pub use polyhedral_harmonic_engine::{
    PolyhedralHarmonicEngine, PolyhedralResonanceReport, U57LayerDetails,
};
pub use riemannian_mercy_manifold::{GeometricTransportResult, RiemannianMercyManifold};
pub use tolc_seven_mercy_gates::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedAdapter {
        name: &'static str,
        valence: f64,
        coherence: GodlyIntelligenceCoherence,
    }

    impl RaThorSystemAdapter for FixedAdapter {
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
            self.coherence.clone()
        }

        fn apply_epigenetic_blessing(&mut self, _blessing: EpigeneticBlessing) {}
    }

    #[test]
    fn initialize_root_tolc_21_is_a_ternary_tree_of_40_shards() {
        let mut engine = FractalTopologyEngine::new();
        engine.initialize_root(21);
        assert_eq!(engine.shard_count(), 40);
        assert_eq!(engine.current_max_depth(), 3);
        assert_eq!(engine.version, "v14.5-fractal-omnimasterpiece");
    }

    #[test]
    fn empty_input_tolc_scores_are_the_source_formula() {
        assert_eq!(tolc_zero_point_resonance(""), 0.95);
        assert_eq!(
            tolc_zero_point_resonance("tolc"),
            0.95 + (1.0 / 10.0) * 0.05
        );
        let scores = project_through_seven_gates("");
        assert_eq!(scores.len(), 7);
        assert_eq!(scores[0], 0.75 * 0.25);
        assert_eq!(scores[1], 0.75 * 0.20);
        assert_eq!(scores[2], 0.75 * 0.18);
        assert_eq!(scores[3], 0.75 * 0.15);
        assert_eq!(scores[4], 0.75 * 0.12);
        assert_eq!(scores[5], 0.75 * 0.05);
        assert_eq!(scores[6], 0.75 * 0.05);
        let result = compute_tolc_valence("");
        assert!(!result.is_passed());
        assert_eq!(result.total_valence(), 0.95 * 0.75);
        let TOLCValenceResult::Veto {
            threshold,
            failed_gates,
            total_valence,
            tolc_resonance,
            ..
        } = result
        else {
            panic!("empty input takes the veto branch");
        };
        assert_eq!(threshold, 0.999_999_9);
        assert_eq!(failed_gates, vec![1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(tolc_resonance, 0.95);
        assert_eq!(total_valence, 0.95 * 0.75);
    }

    #[test]
    fn valence_min_blessing_impacts_and_dual_names_are_source_literals() {
        assert_eq!(Valence::MIN, 0.9999999);
        let blessing = EpigeneticBlessing::new("RadicalLove", 1.0, "lattice");
        assert_eq!(blessing.evolution_impact, 1.0 * 0.6);
        assert_eq!(blessing.mercy_impact, 1.0 * 0.3);
        assert_eq!(blessing.tolc_impact, 0.02);
        assert_eq!(blessing.blessing_type, "RadicalLove");
        assert_eq!(
            polyhedral_harmonic_engine::get_dual_platonic("Cube"),
            Some("Octahedron")
        );
        assert_eq!(
            polyhedral_harmonic_engine::get_dual_platonic("Tetrahedron"),
            Some("Tetrahedron")
        );
        assert_eq!(
            polyhedral_harmonic_engine::get_dual_platonic("missing"),
            None
        );
        assert_eq!(
            RiemannianMercyManifold::new().current_zeta_critical_line_bound(),
            0.6725
        );
    }

    #[test]
    fn christoffel_entry_matches_the_average_times_point_one() {
        let symbols = riemannian_manifold::compute_christoffel_symbols(&[2.0, 4.0]);
        assert_eq!(symbols.len(), 2);
        assert_eq!(symbols[0].len(), 2);
        assert_eq!(symbols[0][0].len(), 2);
        assert_eq!(symbols[0][1][0], (4.0 + 2.0 + 2.0) / 3.0 * 0.1);
        assert_eq!(symbols[1][1][1], (4.0 + 4.0 + 4.0) / 3.0 * 0.1);
    }

    #[tokio::test]
    async fn empty_cycle_uses_the_source_average_and_a_low_adapter_is_vetoed() {
        let mut orchestrator = QuantumSwarmOrchestrator::new();
        assert_eq!(orchestrator.registered_system_count(), 0);
        assert_eq!(orchestrator.version, "v14.0-one-organism");
        let empty = orchestrator
            .run_one_organism_cycle(OneOrganismContext {
                cycle_id: 7,
                tolc_order: 8,
                base_valence: Valence(0.5),
                patsagi_insight: None,
            })
            .await
            .expect("empty adapter set");
        assert_eq!(empty.cycle_id, 7);
        assert_eq!(empty.average_valence, 1.0);
        assert_eq!(empty.active_systems.len(), 0);
        assert_eq!(empty.overall_coherence.precision, 0.0);
        assert_eq!(empty.overall_coherence.resilience, 0.0);
        assert_eq!(empty.overall_coherence.flow_stability, 0.0);
        assert_eq!(empty.overall_coherence.harmonic_alignment, 0.0);
        assert_eq!(empty.recommended_actions.len(), 3);

        orchestrator.register_adapter(Box::new(FixedAdapter {
            name: "low",
            valence: 0.5,
            coherence: GodlyIntelligenceCoherence {
                precision: 1.0,
                resilience: 0.0,
                flow_stability: 0.0,
                harmonic_alignment: 0.0,
            },
        }));
        let veto = orchestrator
            .run_one_organism_cycle(OneOrganismContext {
                cycle_id: 8,
                tolc_order: 8,
                base_valence: Valence(0.5),
                patsagi_insight: None,
            })
            .await;
        match veto {
            Err(MercyError::ValenceBelowThreshold(value)) => assert_eq!(value, 0.5),
            Ok(_) => panic!("valence 0.5 is below Valence::MIN"),
            Err(other) => panic!("unexpected error: {other}"),
        }

        let mut passing = QuantumSwarmOrchestrator::new();
        let coherence = GodlyIntelligenceCoherence {
            precision: 1.5,
            resilience: 0.25,
            flow_stability: 0.125,
            harmonic_alignment: 4.0,
        };
        passing.register_adapter(Box::new(FixedAdapter {
            name: "a",
            valence: 1.0,
            coherence: coherence.clone(),
        }));
        passing.register_adapter(Box::new(FixedAdapter {
            name: "b",
            valence: 1.0,
            coherence: coherence.clone(),
        }));
        assert_eq!(passing.registered_system_count(), 2);
        let insight = passing
            .run_one_organism_cycle(OneOrganismContext {
                cycle_id: 9,
                tolc_order: 8,
                base_valence: Valence(1.0),
                patsagi_insight: None,
            })
            .await
            .expect("average 1.0 is at least Valence::MIN");
        assert_eq!(insight.average_valence, 1.0);
        assert_eq!(insight.active_systems.len(), 2);
        assert_eq!(insight.overall_coherence.precision, 3.0);
        assert_eq!(insight.overall_coherence.resilience, 0.5);
        assert_eq!(insight.overall_coherence.flow_stability, 0.25);
        assert_eq!(insight.overall_coherence.harmonic_alignment, 8.0);
    }
}
