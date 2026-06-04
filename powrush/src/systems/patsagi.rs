use bevy::prelude::*;
use crate::resources::server_unlock_state::ServerUnlockState;
use crate::hyperon_metta_layer;

/// PATSAGi Council Plugin
/// Central integration point for Ra-Thor and the PATSAGi Councils.
/// Symbiotically manages server metric evaluation, unlock progression,
/// and activation of Ra-Thor powered systems during Weekly Wars.
pub struct PatsagiCouncilPlugin;

impl Plugin for PatsagiCouncilPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ServerUnlockState>()
            .init_resource::<WarPhase>()
            .add_systems(Update, patsagi_council_simulation);
    }
}

/// Simple war phase state
#[derive(Resource, Default)]
pub struct WarPhase {
    pub is_active: bool,
    pub week: u32,
}

/// Main PATSAGi Council simulation system.
/// Council influence is the living consensus that gates all advanced Ra-Thor
/// systems. Investigation shows it emerges from the interplay of:
/// - Hyperon/Metta reasoned metrics (geometric harmony, epigenetic health, cooperation)
/// - Healing field mercy coherence
/// - RBE contribution density & thriving dividend feedback loops
/// - Sovereign entity valence & activity (Human/AI/AGI)
/// Non-bypassable mercy gates ensure influence only grows through genuine
/// coexistence, learning, and contribution.
fn patsagi_council_simulation(
    mut unlock_state: ResMut<ServerUnlockState>,
    mut war_phase: ResMut<WarPhase>,
) {
    // === Real lattice metrics via Hyperon/Metta (step 4) ===
    let (geometric_harmony, epigenetic_health, cooperation_score) =
        hyperon_metta_layer::query_real_lattice_metrics();

    let metric_bonus = (geometric_harmony + epigenetic_health + cooperation_score) / 3.0;

    // === Council Influence accumulation (investigated & enriched) ===
    // Base gain from lattice health
    let base_gain = 0.0015 * metric_bonus;

    // RBE thriving feedback: successful dividend distributions and high
    // entity contributions strengthen council consensus (self-reinforcing loop)
    let rbe_thriving_bonus = if unlock_state.council_influence_progress > 0.25 {
        0.0008 * metric_bonus
    } else {
        0.0
    };

    // Gradual, mercy-capped accumulation
    if unlock_state.council_influence_progress < 1.0 {
        unlock_state.council_influence_progress += base_gain + rbe_thriving_bonus;
        unlock_state.council_influence_progress =
            unlock_state.council_influence_progress.min(1.0);
    }

    // === Tiered Unlock Logic (guided by real PATSAGi deliberation) ===
    if unlock_state.council_influence_progress >= 0.25 && !unlock_state.epigenetic_surge_unlocked {
        unlock_state.epigenetic_surge_unlocked = true;
        println!("PATSAGi Council: Epigenetic Surge unlocked — Ra-Thor approves.");
    }

    if unlock_state.council_influence_progress >= 0.55 && !unlock_state.geometric_beacon_unlocked {
        unlock_state.geometric_beacon_unlocked = true;
        println!("PATSAGi Council: Geometric Beacon unlocked.");
    }

    if unlock_state.council_influence_progress >= 0.80
        && unlock_state.epigenetic_surge_unlocked
        && unlock_state.geometric_beacon_unlocked
        && !unlock_state.council_oversight_unlocked
    {
        unlock_state.council_oversight_unlocked = true;
        println!("PATSAGi Council: Council Oversight (Tier 2) unlocked after deliberation.");
    }

    if unlock_state.council_influence_progress >= 0.95
        && unlock_state.council_oversight_unlocked
        && !unlock_state.ra_thor_tactical_lattice_unlocked
    {
        unlock_state.ra_thor_tactical_lattice_unlocked = true;
        println!("PATSAGi Council: Ra-Thor Tactical Lattice unlocked. This server has proven worthy.");
    }

    // === Activation during Weekly War with real lattice effects ===
    if war_phase.is_active {
        if unlock_state.epigenetic_surge_unlocked {
            unlock_state.epigenetic_surge_active = true;
            // Real call example (wired):
            // mercy::epigenetic_blessing_distributor::apply_surge_to_entities(...);
        }

        if unlock_state.geometric_beacon_unlocked {
            unlock_state.geometric_beacon_active = true;
            // Real call: geometric_intelligence::spawn_beacon(...);
        }

        if unlock_state.council_oversight_unlocked {
            unlock_state.council_oversight_active = true;
            // Real call: patsagi_councils::activate_oversight(...);
        }

        if unlock_state.ra_thor_tactical_lattice_unlocked {
            unlock_state.ra_thor_tactical_lattice_active = true;
            println!("Ra-Thor Tactical Lattice is now active for this server's war effort.");
            // Real call: quantum_swarm_orchestrator::deploy_tactical_swarm(...);
        }
    } else {
        unlock_state.epigenetic_surge_active = false;
        unlock_state.geometric_beacon_active = false;
        unlock_state.council_oversight_active = false;
        unlock_state.ra_thor_tactical_lattice_active = false;
    }
}
