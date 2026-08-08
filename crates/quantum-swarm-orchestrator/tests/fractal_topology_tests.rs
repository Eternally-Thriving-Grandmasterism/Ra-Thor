// tests/fractal_topology_tests.rs
// Basic unit tests for FractalTopologyEngine v14.5

use ra_thor_quantum_swarm_orchestrator::fractal_topology_engine::FractalTopologyEngine;
use ra_thor_quantum_swarm_orchestrator::polyhedral_harmonic_engine::PolyhedralHarmonicEngine;
use ra_thor_quantum_swarm_orchestrator::riemannian_mercy_manifold::RiemannianMercyManifold;

#[test]
fn test_fractal_engine_initializes_root() {
    let mut engine = FractalTopologyEngine::new();
    engine.initialize_root(21);

    assert!(engine.shard_count() > 1);
    assert!(engine.current_max_depth() >= 2);
}

#[test]
fn test_higher_tolc_unlocks_deeper_recursion() {
    let mut low = FractalTopologyEngine::new();
    low.initialize_root(10);

    let mut high = FractalTopologyEngine::new();
    high.initialize_root(160);

    assert!(high.current_max_depth() > low.current_max_depth());
    assert!(high.shard_count() > low.shard_count());
}

#[test]
fn test_fractal_cycle_produces_report() {
    let mut engine = FractalTopologyEngine::new();
    let poly = PolyhedralHarmonicEngine::new();
    let riemann = RiemannianMercyManifold::new();

    let report = engine.run_fractal_cycle(55, &poly, &riemann, 0.95);

    assert!(report.total_shards > 0);
    assert!(report.active_shards > 0);
    assert!(report.average_valence >= 0.999999);
    assert!(!report.suggested_blessings.is_empty());
}
