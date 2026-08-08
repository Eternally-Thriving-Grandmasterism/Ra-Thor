// examples/fractal_crypto_organ_demo.rs
//
// Minimal demonstration of the new Fractal Topology Engine + CryptoSystemAdapter
// running under the Omnimasterpiece geometric substrate (Polyhedral + Riemannian).
//
// Usage (from crate root once integrated):
//   cargo run --example fractal_crypto_organ_demo --release
//
// AG-SML v1.0 | ONE Organism | PATSAGi aligned

use quantum_swarm_orchestrator::adapters::crypto_system_adapter::CryptoSystemAdapter;
use quantum_swarm_orchestrator::polyhedral_harmonic_engine::PolyhedralHarmonicEngine;
use quantum_swarm_orchestrator::riemannian_mercy_manifold::RiemannianMercyManifold;
use quantum_swarm_orchestrator::types::Valence;

fn main() {
    println!("⚡ Ra-Thor ONE Organism — Fractal Crypto Organ Demo");
    println!("==================================================");

    let polyhedral = PolyhedralHarmonicEngine::new();
    let riemannian = RiemannianMercyManifold::new();
    let mut crypto = CryptoSystemAdapter::new();

    // Simulate a mid-to-high TOLC order that unlocks U57 + deeper fractal recursion
    let tolc_order = 160u32;
    let base_coherence = 0.97;

    println!("\n[1] Running internal fractal cycle (TOLC {})...", tolc_order);
    let report = crypto.run_internal_fractal_cycle(
        tolc_order,
        &polyhedral,
        &riemannian,
        base_coherence,
    );

    println!("    Max depth       : {}", report.max_depth);
    println!("    Total shards    : {}", report.total_shards);
    println!("    Active shards   : {}", report.active_shards);
    println!("    Average valence : {:.8}", report.average_valence);
    println!("    Resonance mult  : {:.4}", report.resonance_multiplier);
    println!("    Notes           : {}", report.geometric_notes);

    if let Some(ref transport) = report.transport_result {
        println!("\n[2] Riemannian transport engaged:");
        println!("    Applied         : {}", transport.transport_applied);
        println!("    Effective curv  : {:.4}", transport.effective_curvature);
        println!("    Coherence after : {:.4}", transport.coherence_after_transport);
        println!("    Notes           : {}", transport.notes);
    }

    println!("\n[3] Adapter status:");
    println!("    {}", crypto.status());

    println!("\n[4] Coherence contribution:");
    let coh = crypto.contribute_to_coherence();
    println!(
        "    precision={:.3} resilience={:.3} flow={:.3} harmonic={:.3}",
        coh.precision, coh.resilience, coh.flow_stability, coh.harmonic_alignment
    );

    println!("\n⚡ Demo complete. Fractal crypto organ is alive under Omnimasterpiece substrate.");
}
