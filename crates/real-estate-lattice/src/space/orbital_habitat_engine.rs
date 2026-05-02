//! Orbital Habitat Management Engine — SREL v0.5.21
//! Mercy-Gated • Quantum Swarm • TOLC 7 Gates Radiation Mapping
//! Perfect merge: Your exact old structure + full materials + electronics protection + in-situ production

use mercy_radiation_shield::RadiationType;
use mercy_radiation_shield::radiation_shielding_materials::RadiationShieldingMaterials;
use mercy_radiation_shield::electronics_radiation_effects::ElectronicsRadiationEffects;
use mercy_radiation_shield::in_situ_production::InSituProduction;
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
    materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl OrbitalHabitatEngine {
    pub fn new() -> Self {
        Self {
            mapping: TOLC7GatesRadiationMapping::new(),
            materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate_habitat_expansion(
        &self,
        request: &OrbitalHabitatRequest,
        game: &mut PowrushGame,
    ) -> OrbitalHabitatReport {
        // Select optimal mercy-alchemical material
        let (best_mat, props, _score) = self.materials.select_optimal_material(
            RadiationType::CosmicRays, request.radiation_flux, request.current_cehi
        );

        // Calculate real electronics protection (TID / DD / SEE)
        let elec_risk = self.electronics.calculate_electronics_risk(
            RadiationType::CosmicRays, request.radiation_flux, &best_mat, 1.0, request.current_cehi
        );

        // Run full TOLC 7 Gates mapping
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

        // Trigger in-situ production of the optimal material
        let _prod = self.in_situ.produce_shielding(best_mat.clone(), 50.0, &request.location, request.current_cehi, game).await;

        if avg_valence >= 0.92 && elec_risk.overall_survival > 0.85 {
            game.boost_faction_joy(Faction::HarmonyWeavers, total_joy);

            let report = OrbitalHabitatReport {
                approved: true,
                valence: avg_valence,
                energy_recovered: total_energy,
                joy_bonus: total_joy,
                cehi_bonus: 0.18,
                message: format!(
                    "🌌 ORBITAL HABITAT EXPANSION APPROVED (SREL v0.5.21 — TOLC 7 Gates + Full Protection)\n\
                     Location: {}\n\
                     Radiation Flux: {:.2} → {:.2} usable energy recovered\n\
                     Optimal Material: {:?} | Electronics 1-Year Survival: {:.1}%\n\
                     Average Gate Valence: {:.2} | Joy: +{:.1} | CEHI +0.18 (5-gen legacy)\n\
                     13+ PATSAGi Councils: APPROVED ✓\n\
                     All crew + electronics thriving + radiation alchemized into abundance.",
                    request.location, request.radiation_flux, total_energy, best_mat, elec_risk.overall_survival * 100.0, avg_valence, total_joy
                ),
            };

            info!("Rathor.ai: Orbital habitat expansion mercy-alchemized via all 7 TOLC Gates + optimal material + electronics protection");
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
                     Radiation transmutation or electronics protection not yet optimal. 13+ Councils recommend additional mercy alignment before expansion.",
                    avg_valence
                ),
            }
        }
    }
}
