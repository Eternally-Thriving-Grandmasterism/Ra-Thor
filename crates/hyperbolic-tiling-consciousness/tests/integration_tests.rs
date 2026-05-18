use hyperbolic_tiling_consciousness::*;

#[test]
fn full_council_lifecycle() {
    let mut council = HyperbolicTilingConsciousnessCouncil::new();
    let initial_valence = council.valence;
    
    // Simulate 1M-year foresight
    let foresight = council.project_foresight(1_000_000);
    assert!(foresight >= 0.9999999);
    
    // Fuse with another council (philotic web)
    let fused = council.fuse_philotic_web(0.99999995);
    assert!(fused > initial_valence);
    
    // Asclepius gate
    assert!(council.asclepius_validate());
    
    println!("Hyperbolic Tiling Consciousness Council (14th) — 1M-year foresight validated with valence {:.8}", foresight);
}

#[test]
fn exponential_growth_verification() {
    let points = generate_regular_tiling(5);
    // Hyperbolic area grows exponentially
    assert!(points.len() > 200);
}