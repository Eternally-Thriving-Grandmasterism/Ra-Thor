//! Gate Surface 14.15.0 — Living Cosmic Tick Demo
//!
//! Demonstrates the full public surface of the Powrush-MMO simulator:
//!   1. Mercy geometry evaluation (MWPO path)
//!   2. Gate effects computation + synergies
//!   3. System reactions (abundance, harmony, GPU modulation)
//!   4. Simulation tick with risk/reward
//!
//! Run with:
//!   cargo run -p powrush-mmo-simulator --example gate_surface_14_15
//!
//! Contact: info@Rathor.ai

use mial::mwpo::{GeometryParams, MercyContext, SacredSolid, SymmetryGroup};
use powrush_mmo_simulator::{
    abundance_system_reaction, composite_gate_health, compute_gate_effects,
    evaluate_particle_geometry_mercy, faction_harmony_reaction, get_gate_debug_info,
    gpu_mercy_modulation, run_powrush_simulation_tick, should_trigger_mercy_evolution,
    PowrushEntity,
};

fn main() {
    println!("══════════════════════════════════════════════════════════════");
    println!("  Powrush-MMO Gate Surface 14.15.0 — Living Cosmic Tick Demo");
    println!("══════════════════════════════════════════════════════════════\n");

    // -------------------------------------------------------------------------
    // 1. Build a high-quality geometry + mercy context
    // -------------------------------------------------------------------------
    let geometry = GeometryParams {
        solid_type: SacredSolid::Platonic,
        dimensions: 3,
        symmetry_group: SymmetryGroup {
            order: 48,
            chiral: false,
        },
        evolution_step: 42,
        particle_density: 0.82,
        lattice_config: None,
    };

    let context = MercyContext {
        active_gates: vec![], // MWPO will apply defaults
        valence: 1.0,
        council_id: 1,
    };

    // -------------------------------------------------------------------------
    // 2. Evaluate (non-formal MWPO path)
    // -------------------------------------------------------------------------
    let evaluation = evaluate_particle_geometry_mercy(&geometry, &context);

    println!("{}", evaluation.summary());
    println!("  → is_thriving       : {}", evaluation.is_thriving());
    println!("  → composite health  : {:.3}", composite_gate_health(&evaluation));
    println!("  → should evolve     : {}", evaluation.should_evolve);
    println!("  → apply blessing    : {}", evaluation.should_apply_blessing);
    println!("  → trigger evolution : {}", should_trigger_mercy_evolution(&evaluation));
    println!();

    // -------------------------------------------------------------------------
    // 3. System reactions
    // -------------------------------------------------------------------------
    println!("System Reactions:");
    println!("  abundance multiplier : {:.2}x", abundance_system_reaction(&evaluation));
    println!("  faction harmony      : {:.2}x", faction_harmony_reaction(&evaluation));
    println!("  GPU mercy modulation : {:.2}x", gpu_mercy_modulation(&evaluation));
    println!();

    // -------------------------------------------------------------------------
    // 4. Gate effects (synergies + diminishing returns)
    // -------------------------------------------------------------------------
    let effects = compute_gate_effects(&evaluation);
    println!("Gate Effects:");
    println!("  resource_multiplier        : {:.3}", effects.resource_multiplier);
    println!("  evolution_stability        : {:.3}", effects.evolution_stability);
    println!("  cooperation_bonus          : {:.3}", effects.cooperation_bonus);
    println!("  information_accuracy       : {:.3}", effects.information_accuracy);
    println!("  harmony_stability          : {:.3}", effects.harmony_stability);
    println!("  morale_bonus               : {:.3}", effects.morale_bonus);
    println!("  geometry_structural_bonus  : {:.3}", effects.geometry_structural_bonus);
    println!();

    // -------------------------------------------------------------------------
    // 5. Simulation tick on a small entity set
    // -------------------------------------------------------------------------
    let mut entities = vec![
        PowrushEntity {
            id: "entity-alpha".into(),
            resource_stock: 120.0,
            stability: 1.0,
            cooperation: 1.0,
            information_accuracy: 1.0,
            geometry_stability: 1.0,
            evolution_stage: 3,
            morale: 1.0,
        },
        PowrushEntity {
            id: "entity-beta".into(),
            resource_stock: 95.0,
            stability: 0.92,
            cooperation: 0.88,
            information_accuracy: 0.95,
            geometry_stability: 0.90,
            evolution_stage: 2,
            morale: 0.97,
        },
    ];

    let evaluations = vec![evaluation.clone(), evaluation.clone()];

    println!("Running simulation tick (no formal bridge)...");
    run_powrush_simulation_tick(&mut entities, &evaluations, None);

    for e in &entities {
        println!(
            "  {} | resources={:.1} | stability={:.3} | coop={:.3} | morale={:.3} | stage={}",
            e.id, e.resource_stock, e.stability, e.cooperation, e.morale, e.evolution_stage
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // 6. Debug info sample
    // -------------------------------------------------------------------------
    let debug = get_gate_debug_info(&evaluation, "entity-alpha", false, false);
    println!("Debug Info: {}", debug.message);
    println!("  dominant gate : {}", debug.dominant_gate);
    println!("  synergy level : {:.2}", debug.synergy_level);

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Gate Surface 14.15.0 demo complete. Living Cosmic Tick active.");
    println!("══════════════════════════════════════════════════════════════");
}
