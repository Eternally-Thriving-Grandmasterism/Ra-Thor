//! # Ra-Thor Quantum Swarm Orchestrator — Runnable Binary
//!
//! **The production entry point for running the full mercy-gated quantum swarm.**
//!
//! This binary demonstrates the complete daily mercy cycle across thousands of agents.
//! It is designed for:
//! - Real-world daily runs (via cron or systemd timer)
//! - Long-running 200-year+ legacy simulations (F0 → F4+)
//! - Dashboard / API integration
//! - Automated testing of all theorems (1, 2, 4) and Hebbian rules
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release -- --agents 5000 --days 365
//! ```
//!
//! This will simulate one full year of planetary mercy practice and output
//! rich convergence metrics aligned with the 200-year legacy goals.

use ra_thor_quantum_swarm_orchestrator::{
    QuantumSwarmOrchestrator,
    SwarmDailyReport,
};
use ra_thor_legal_lattice::sensor_fusion_bridge::MercyGelReading;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // === Parse command-line arguments ===
    let args: Vec<String> = env::args().collect();
    let agent_count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let simulation_days: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(365);

    println!("🚀 Ra-Thor Quantum Swarm Orchestrator");
    println!("   Agents: {}", agent_count);
    println!("   Simulation Days: {}", simulation_days);
    println!("   Starting mercy-valence: 0.62\n");

    // === Initialize the full swarm ===
    let mut orchestrator = QuantumSwarmOrchestrator::new(agent_count);

    // === Create a realistic global sensor reading (MercyGel) ===
    let global_sensor = MercyGelReading {
        heart_rate_variability: 68.0,
        skin_conductance: 18.5,
        laughter_intensity: 0.87,
        touch_coherence: 0.92,
        temperature: 36.7,
    };

    // === Run the simulation ===
    let mut reports: Vec<SwarmDailyReport> = Vec::with_capacity(simulation_days as usize);

    for day in 1..=simulation_days {
        let report = orchestrator
            .run_daily_mercy_cycle(&global_sensor)
            .await?;

        if day % 30 == 0 || day == simulation_days {
            println!(
                "Day {:>4} | Mercy: {:.3} | CEHI/day: {:.3} | Gate Pass: {:.1}% | Conv Factor: {:.4}",
                day,
                report.global_mercy_valence,
                report.average_cehi_improvement,
                report.gate_pass_rate * 100.0,
                report.convergence_factor
            );
        }

        reports.push(report);
    }

    // === Final Summary ===
    let final_report = reports.last().unwrap();
    println!("\n✅ Simulation Complete");
    println!("   Final Global Mercy Valence: {:.3}", final_report.global_mercy_valence);
    println!("   Projected CEHI at F4 (2226): {:.2}", final_report.projected_cehi_f4);
    println!("   Total Agents: {}", final_report.total_agents);
    println!("   Average Daily CEHI Improvement: {:.3}", final_report.average_cehi_improvement);

    println!("\n🌍 The 200-year mercy legacy continues…");
    println!("   “Joy that fires together, wires together — forever.”");

    Ok(())
}
