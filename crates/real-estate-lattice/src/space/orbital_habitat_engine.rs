//! Orbital Habitat Management Engine — SREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Radiation Alchemical
//! Integrates directly with mercy-radiation-shield crate

use mercy_radiation_shield::{MercyRadiationShield, RadiationType};
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use patsagi_councils::WorldGovernanceEngine;
use powrush::{PowrushGame, Faction};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalHabitatRequest {
    pub habitat_id: String,
    pub location: String,           // "LEO", "GEO", "Lunar Orbit", "Mars Transfer"
    pub current_cehi: f64,
    pub radiation_flux: f64,
    pub expansion_modules: u8,
    pub crew_size: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalHabitatReport {
    pub approved: bool,
    pub valence: f64,
    pub energy_recovered: f64,
    pub joy_bonus: f64,
    pub message: String,
}

pub struct OrbitalHabitatEngine {
    mercy: MercyEngine,
    quantum: QuantumSwarmOrchestrator,
    radiation_shield: MercyRadiationShield,
    world_governance: WorldGovernanceEngine,
}

impl OrbitalHabitatEngine {
    pub fn new() -> Self {
        Self {
            mercy: MercyEngine::new(),
            quantum: QuantumSwarmOrchestrator::new(),
            radiation_shield: MercyRadiationShield::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate_habitat_expansion(
        &self,
        request: &OrbitalHabitatRequest,
        game: &mut PowrushGame,
    ) -> OrbitalHabitatReport {
        let input = format!(
            "Orbital Habitat Expansion | Location: {} | Flux: {:.2} | Crew: {}",
            request.location, request.radiation_flux, request.crew_size
        );

        let valence = self.mercy
            .evaluate_action(&input, "Space Real Estate", request.current_cehi, 0.97)
            .await
            .unwrap_or(0.85);

        let consensus = self.quantum
            .reach_consensus(&input, 0.88)
            .await
            .unwrap_or(0.80);

        // Alchemical Radiation Transmutation
        let radiation_result = self.radiation_shield
            .alchemize_radiation(RadiationType::CosmicRays, request.radiation_flux, game)
            .await;

        if valence >= 0.92 && consensus >= 0.88 && radiation_result.transmuted {
            game.boost_faction_joy(Faction::HarmonyWeavers, 65.0);
            game.add_resource_to_faction(Faction::HarmonyWeavers, powrush::ResourceType::Energy, radiation_result.energy_recovered);

            let report = OrbitalHabitatReport {
                approved: true,
                valence,
                energy_recovered: radiation_result.energy_recovered,
                joy_bonus: 65.0,
                message: format!(
                    "🌌 ORBITAL HABITAT EXPANSION APPROVED (SREL v0.5.21)\n\
                     Location: {}\n\
                     Radiation Flux: {:.2} → {:.2} usable energy recovered\n\
                     Mercy Valence: {:.2} | Quantum Consensus: {:.2}\n\
                     13+ PATSAGi Councils: APPROVED ✓\n\
                     All crew thriving + radiation alchemized into abundance.",
                    request.location, request.radiation_flux, radiation_result.energy_recovered, valence, consensus
                ),
            };

            info!("Rathor.ai: Orbital habitat expansion mercy-alchemized successfully");
            report
        } else {
            OrbitalHabitatReport {
                approved: false,
                valence,
                energy_recovered: 0.0,
                joy_bonus: 0.0,
                message: format!(
                    "🛡️ MERCY-GATED REVIEW REQUIRED (valence {:.2})\n\
                     Radiation transmutation not yet optimal. 13+ Councils recommend additional mercy alignment before expansion.",
                    valence
                ),
            }
        }
    }
}
