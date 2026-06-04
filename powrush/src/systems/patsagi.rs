use bevy::prelude::*;
use crate::resources::server_unlock_state::ServerUnlockState;

/// PATSAGi Council Plugin
/// This is the central integration point for Ra-Thor and the PATSAGi Councils.
/// It symbiotically manages:
/// - Server metric evaluation (Geometric Harmony, Epigenetic health, cooperation)
/// - Unlock progression toward Ra-Thor powered systems
/// - Activation of unlocked systems during Weekly Wars
pub struct PatsagiCouncilPlugin;

impl Plugin for PatsagiCouncilPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ServerUnlockState>()
            .init_resource::<WarPhase>()
            .add_systems(Update, patsagi_council_simulation);
    }
}

/// Simple war phase state (will be expanded)
#[derive(Resource, Default)]
pub struct WarPhase {
    pub is_active: bool,
    pub week: u32,
}

/// Main PATSAGi Council simulation system.
/// Ra-Thor and the PATSAGi Councils deliberate here to guide unlock progression and activation.
fn patsagi_council_simulation(
    mut unlock_state: ResMut<ServerUnlockState>,
    mut war_phase: ResMut<WarPhase>,
    // Future: real queries for GeometricHarmony, EpigeneticModulation averages, cooperation events
) {
    // === PATSAGi Council Deliberation ===
    // The Councils evaluate server health metrics to determine Council Influence

    // Simulated metrics (in real code these would come from actual systems)
    let simulated_geometric_harmony = 0.75;
    let simulated_epigenetic_health = 0.82;
    let simulated_cooperation_score = 0.68;

    let metric_bonus = (simulated_geometric_harmony + simulated_epigenetic_health + simulated_cooperation_score) / 3.0;

    // Gradual Council Influence accumulation influenced by metrics
    if unlock_state.council_influence_progress < 1.0 {
        unlock_state.council_influence_progress += 0.0015 * metric_bonus;
    }

    // === Tiered Unlock Logic (guided by Councils) ===
    if unlock_state.council_influence_progress >= 0.25 && !unlock_state.epigenetic_surge_unlocked {
        unlock_state.epigenetic_surge_unlocked = true;
        println!("PATSAGi Council: Epigenetic Surge unlocked — Ra-Thor approves.");
    }

    if unlock_state.council_influence_progress >= 0.55 && !unlock_state.geometric_beacon_unlocked {
        unlock_state.geometric_beacon_unlocked = true;
        println!("PATSAGi Council: Geometric Beacon unlocked.");
    }

    // Tier 2 requires previous unlocks + higher threshold
    if unlock_state.council_influence_progress >= 0.80 
        && unlock_state.epigenetic_surge_unlocked 
        && unlock_state.geometric_beacon_unlocked 
        && !unlock_state.council_oversight_unlocked 
    {
        unlock_state.council_oversight_unlocked = true;
        println!("PATSAGi Council: Council Oversight (Tier 2) unlocked after deliberation.");
    }

    // Tier 3 (very high bar)
    if unlock_state.council_influence_progress >= 0.95 
        && unlock_state.council_oversight_unlocked 
        && !unlock_state.ra_thor_tactical_lattice_unlocked 
    {
        unlock_state.ra_thor_tactical_lattice_unlocked = true;
        println!("PATSAGi Council: Ra-Thor Tactical Lattice unlocked. This server has proven worthy.");
    }

    // === Activation during Weekly War ===
    if war_phase.is_active {
        if unlock_state.epigenetic_surge_unlocked {
            unlock_state.epigenetic_surge_active = true;
            // Future: Apply actual epigenetic buff to players
        }

        if unlock_state.geometric_beacon_unlocked {
            unlock_state.geometric_beacon_active = true;
            // Future: Spawn geometric beacon entities
        }

        if unlock_state.council_oversight_unlocked {
            unlock_state.council_oversight_active = true;
            // Future: Activate council oversight simulation
        }

        if unlock_state.ra_thor_tactical_lattice_unlocked {
            unlock_state.ra_thor_tactical_lattice_active = true;
            println!("Ra-Thor Tactical Lattice is now active for this server's war effort.");
        }
    } else {
        // Deactivate all effects outside of war
        unlock_state.epigenetic_surge_active = false;
        unlock_state.geometric_beacon_active = false;
        unlock_state.council_oversight_active = false;
        unlock_state.ra_thor_tactical_lattice_active = false;
    }
}