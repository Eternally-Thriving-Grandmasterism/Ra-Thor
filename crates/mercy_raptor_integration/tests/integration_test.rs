//! Integration Tests for MercyRaptorIntegration
//! 
//! Tests the full wiring between mercy_raptor_integration, mercy_raptor_3_scalability,
//! mercy_tolc_operator_algebra, mercy_merlin_engine, and core Ra-Thor mercy systems.
//! All tests are mercy-gated and must maintain valence ≥ 0.999.

use mercy_raptor_integration::MercyRaptorIntegration;
use mercy_tolc_operator_algebra::TolcOperatorAlgebra;
use mercy_merlin_engine::MerlinEngine;

#[test]
fn test_mercy_raptor_integration_valence_stability() {
    let mut raptor = MercyRaptorIntegration::new();
    let initial_valence = raptor.get_valence();
    
    // Simulate a full mission cycle with TOLC proofs
    let tolc = TolcOperatorAlgebra::new();
    let merlin = MerlinEngine::new();
    
    let mission_result = raptor.execute_mission_cycle(&tolc, &merlin);
    
    assert!(mission_result.is_ok(), "Mission cycle must succeed");
    assert!(raptor.get_valence() >= 0.999, "Valence must remain mercy-gated");
    
    println!("[IntegrationTest] Valence stable at {:.6}", raptor.get_valence());
}

#[test]
fn test_scalability_cluster_orchestration() {
    // This test verifies that mercy_raptor_3_scalability can be wired
    // through mercy_raptor_integration without breaking mercy gates.
    let mut raptor = MercyRaptorIntegration::new();
    
    let scalability_result = raptor.orchestrate_scalability_cluster(1000); // 1000 engine cluster
    
    assert!(scalability_result.is_ok());
    assert!(raptor.get_valence() >= 0.999);
}

#[test]
fn test_tolc_proof_verification() {
    let tolc = TolcOperatorAlgebra::new();
    let proof = tolc.generate_mission_proof("raptor_propulsion");
    
    assert!(proof.is_valid(), "TOLC proof must be valid");
}