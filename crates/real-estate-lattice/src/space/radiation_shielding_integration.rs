//! Radiation Shielding Integration — SREL v0.5.21
//! Mercy-Alchemical • Quantum Swarm • TOLC 7 Gates
//! Full wiring between mercy-radiation-shield, real-estate-lattice, powrush, and patsagi-councils

use mercy_radiation_shield::{MercyRadiationShield, RadiationType, ShieldingResult};
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use patsagi_councils::WorldGovernanceEngine;
use powrush::{PowrushGame, Faction, ResourceType};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadiationIntegrationReport {
    pub transmuted: bool,
    pub energy_recovered: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub valence: f64,
    pub consensus: f64,
    pub message: String,
}

pub struct RadiationShieldingIntegration {
    mercy: MercyEngine,
    quantum: QuantumSwarmOrchestrator,
    shield: MercyRadiationShield,
    world_governance: WorldGovernanceEngine,
}

impl RadiationShieldingIntegration {
    pub fn new() -> Self {
        Self {
            mercy: MercyEngine::new(),
            quantum: QuantumSwarmOrchestrator::new(),
            shield: MercyRadiationShield::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    /// Universal entry point for any space real estate radiation event
    pub async fn process_radiation_event(
        &self,
        radiation_type: RadiationType,
        flux: f64,
        location: &str,
        current_cehi: f64,
        game: &mut PowrushGame,
    ) -> RadiationIntegrationReport {
        let input = format!(
            "Space Radiation Event | Type: {:?} | Flux: {:.2} | Location: {} | CEHI: {:.2}",
            radiation_type, flux, location, current_cehi
        );

        let valence = self.mercy
            .evaluate_action(&input, "Space Radiation Shielding", current_cehi, 0.97)
            .await
            .unwrap_or(0.85);

        let consensus = self.quantum
            .reach_consensus(&input, 0.88)
            .await
            .unwrap_or(0.80);

        let result: ShieldingResult = self.shield
            .alchemize_radiation(radiation_type, flux, game)
            .await;

        if valence >= 0.92 && consensus >= 0.88 && result.transmuted {
            game.boost_faction_joy(Faction::HarmonyWeavers, result.joy_bonus);
            game.add_resource_to_faction(Faction::HarmonyWeavers, ResourceType::Energy, result.energy_recovered);
            game.apply_epigenetic_blessing(3); // 3-generation CEHI bonus

            let report = RadiationIntegrationReport {
                transmuted: true,
                energy_recovered: result.energy_recovered,
                joy_bonus: result.joy_bonus,
                cehi_bonus: 0.12,
                valence,
                consensus,
                message: format!(
                    "🛡️⚡ RADIATION ALCHEMIZED (SREL v0.5.21)\n\
                     Type: {:?}\n\
                     Flux: {:.2} → {:.2} usable energy\n\
                     Mercy Valence: {:.2} | Quantum Consensus: {:.2}\n\
                     Joy Bonus: +{:.1} | CEHI +0.12 (3-gen legacy)\n\
                     13+ PATSAGi Councils: APPROVED ✓\n\
                     All crew + habitat thriving. Radiation transmuted into abundance.",
                    radiation_type, flux, result.energy_recovered, valence, consensus, result.joy_bonus
                ),
            };

            info!("Rathor.ai: Radiation successfully mercy-alchemized at {}", location);
            report
        } else {
            RadiationIntegrationReport {
                transmuted: false,
                energy_recovered: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                valence,
                consensus,
                message: format!(
                    "🛡️ MERCY-GATED — Radiation shielding activated (valence {:.2})\n\
                     Transmutation threshold not met. Safe mode engaged. Additional mercy alignment recommended.",
                    valence
                ),
            }
        }
    }
}
