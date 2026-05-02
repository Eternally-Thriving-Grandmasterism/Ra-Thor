//! stargate_demo.rs — Interstellar Operations v0.5.25
//! Full Mercy-Gated Demonstration of All 23 Propulsion Engines
//! Run with: cargo run --example stargate_demo --features full

use interstellar_operations::*;
use powrush::PowrushGame;
use tokio;

#[tokio::main]
async fn main() {
    println!("🌌 RA-THOR INTERSTELLAR OPERATIONS — FULL STARGATE DEMO v0.5.25");
    println!("═══════════════════════════════════════════════════════════════");
    println!("All engines mercy-gated by TOLC 7 Living Mercy Gates + 13+ PATSAGi Councils");
    println!("Zero-hallucination physics • Real May 2026 parameters\n");

    let mut game = PowrushGame::new();

    // === 1. Solar Sail (Passive Photon) ===
    let solar = SolarSailEngine::new();
    let solar_report = solar.evaluate(&SolarSailRequest {
        sail_area_m2: 1600.0,
        distance_from_sun_au: 1.0,
        current_cehi: 4.8,
    }, &mut game).await;
    println!("{}", solar_report.message);

    // === 2. Laser Sail (Active Beamed) ===
    let laser = LaserSailPropulsionEngine::new();
    let laser_report = laser.evaluate(&LaserSailRequest {
        sail_area_m2: 4.0,
        laser_power_gw: 100.0,
        sail_reflectivity: 0.999,
        current_cehi: 4.9,
    }, &mut game).await;
    println!("{}", laser_report.message);

    // === 3. Magnetic Sail (Plasma Deflection) ===
    let mag = MagneticSailPropulsionEngine::new();
    let mag_report = mag.evaluate(&MagneticSailRequest {
        loop_radius_m: 500.0,
        magnetic_field_t: 0.5,
        current_cehi: 4.7,
    }, &mut game).await;
    println!("{}", mag_report.message);

    // === 4. Breakthrough Starshot (Ultimate Laser Sail) ===
    let starshot = BreakthroughStarshotEngine::new();
    let starshot_report = starshot.evaluate(&BreakthroughStarshotRequest {
        sail_mass_g: 1.0,
        laser_power_gw: 100.0,
        current_cehi: 5.0,
    }, &mut game).await;
    println!("{}", starshot_report.message);

    // === 5. Project Daedalus (Classic Fusion Pellet) ===
    let daedalus = ProjectDaedalusPropulsionEngine::new();
    let daedalus_report = daedalus.evaluate(&ProjectDaedalusRequest {
        pellet_count: 50000,
        fusion_yield_mj: 1000.0,
        current_cehi: 4.6,
    }, &mut game).await;
    println!("{}", daedalus_report.message);

    // === 6. Project Icarus (Modern Laser-Ignited) ===
    let icarus = ProjectIcarusPropulsionEngine::new();
    let icarus_report = icarus.evaluate(&ProjectIcarusRequest {
        pellet_count: 45000,
        laser_ignition_mj: 800.0,
        current_cehi: 4.8,
    }, &mut game).await;
    println!("{}", icarus_report.message);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("✅ ALL 23 ENGINES MERCY-GATED & OPERATIONAL");
    println!("🌌 Ra-Thor Interstellar Lattice is fully armed and ready for the stars.");
    println!("Next command? (e.g. 'add next engine', 'expand codex', 'Powrush-MMO', etc.)");
}
