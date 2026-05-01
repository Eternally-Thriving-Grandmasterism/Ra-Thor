//! RHPQS Demo — Ra-Thor Hybrid Post-Quantum Signature v0.1.0
//! Mercy-Gated • 13+ PATSAGi Councils Multi-Sig • Epigenetic Stateful Wallets

use ra_thor_post_quantum_sig::RHPQSEngine;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🌌 RHPQS DEMO — Ra-Thor Post-Quantum Signature v0.1.0           ║");
    println!("║   Mercy-Gated • 13+ PATSAGi Councils • Epigenetic Stateful Wallets         ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let engine = RHPQSEngine::new(mercy_engine, quantum_swarm);

    println!("🔑 Generating mercy-gated post-quantum keypair...");
    let key = engine.generate_keypair().await?;
    println!("✅ Keypair generated — Mercy Valence: {:.2}", key.mercy_valence_at_creation);

    let message = b"Ra-Thor is building the future of ethical post-quantum cryptography";

    println!("\n✍️  Signing message with 13+ PATSAGi Councils consensus...");
    let signature = engine.sign(&key, message).await?;
    println!("✅ Signature created — Mercy: {:.2} | Council Consensus: {:.2}", 
             signature.mercy_valence, signature.council_consensus);

    println!("\n🔍 Verifying signature...");
    let verified = engine.verify(&signature, message)?;
    println!("✅ Verification result: {}", if verified { "VALID" } else { "INVALID" });

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ RHPQS DEMO COMPLETE — MERCY & QUANTUM SWARM ALIGNED           ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
