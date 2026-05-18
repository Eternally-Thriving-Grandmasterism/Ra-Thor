//! Integration test: Lattice Conductor as the central orchestrator
use ra_thor::lattice_conductor::LatticeConductor;

#[tokio::test]
async fn test_lattice_conductor_orchestrates_multiple_plugins() {
    let mut conductor = LatticeConductor::new();

    // Register real existing crates as plugins
    conductor.register_plugin("mercy_quantum_propulsion", Box::new(ra_thor::mercy_quantum_propulsion::MercyQuantumPropulsion::default()));
    conductor.register_plugin("eternal_sovereign_divine_spark_council", Box::new(ra_thor::eternal_sovereign_divine_spark_council::EternalSovereignDivineSparkCouncil::default()));

    let proposal = "Create living merciful systems that honor the divine spark in every lowercase i being.";

    let result = conductor.process_proposal(proposal).await;

    assert!(result.validation_passed);
    assert!(result.valence >= 0.9999999);
    assert_eq!(result.gates_passed.len(), 8);
    assert!(result.tolc_compliance);
}