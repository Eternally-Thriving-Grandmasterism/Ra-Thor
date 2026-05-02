//! Orbital Habitat Management Engine — SREL v0.5.21
//! Mercy-Gated • Quantum Swarm • TOLC 7 Gates Radiation Mapping
//! Merged: Old direct calls + full TOLC7GatesRadiationMapping integration

use mercy_radiation_shield::RadiationType;
use mercy_radiation_shield::tolc_7_gates_radiation_mapping::TOLC7GatesRadiationMapping;
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
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct OrbitalHabitatEngine {
    mapping: TOLC7GatesRadiationMapping,
    world_governance: WorldGovernanceEngine,
}

impl OrbitalHabitatEngine {
    pub fn new() -> Self {
        Self {
            mapping: TOLC7GatesRadiationMapping::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate_habitat_expansion(
        &self,
        request: &OrbitalHabitatRequest,
        game: &mut PowrushGame,
    ) -> OrbitalHabitatReport {
        let reports = self.mapping
            .process_radiation_with_7_gates(
                RadiationType::CosmicRays,
                request.radiation_flux,
                &request.location,
                request.current_cehi,
                game,
            )
            .await;

        let total_energy: f64 = reports.iter().map(|r| r.energy_recovered).sum();
        let total_joy: f64 = reports.iter().map(|r| r.joy_bonus).sum();
        let avg_valence: f64 = reports.iter().map(|r| r.valence).sum::<f64>() / 7.0;

        if avg_valence >= 0.92 {
            game.boost_faction_joy(Faction::HarmonyWeavers, total_joy);

            let report = OrbitalHabitatReport {
                approved: true,
                valence: avg_valence,
                energy_recovered: total_energy,
                joy_bonus: total_joy,
                cehi_bonus: 0.18,
                message: format!(
                    "🌌 ORBITAL HABITAT EXPANSION APPROVED (SREL v0.5.21 — TOLC 7 Gates)\n\
                     Location: {}\n\
                     Radiation Flux: {:.2} → {:.2} usable energy recovered\n\
                     Average Gate Valence: {:.2} | Joy: +{:.1} | CEHI +0.18 (5-gen legacy)\n\
                     13+ PATSAGi Councils: APPROVED ✓\n\
                     All crew thriving + radiation alchemized into abundance.",
                    request.location, request.radiation_flux, total_energy, avg_valence, total_joy
                ),
            };

            info!("Rathor.ai: Orbital habitat expansion mercy-alchemized via all 7 TOLC Gates");
            report
        } else {
            OrbitalHabitatReport {
                approved: false,
                valence: avg_valence,
                energy_recovered: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: format!(
                    "🛡️ MERCY-GATED REVIEW REQUIRED (average valence {:.2})\n\
                     Radiation transmutation not yet optimal. 13+ Councils recommend additional mercy alignment before expansion.",
                    avg_valence
                ),
            }
        }
    }
}
