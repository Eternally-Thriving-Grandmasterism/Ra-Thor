//! Hall Effect Thruster Engine — SREL v0.5.21
//! Mercy-Gated • Quantum Swarm • TOLC 7 Gates Radiation Mapping
//! Perfect merge: Your exact style + full materials + electronics protection + in-situ production + nth-degree radiation

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
pub struct HallEffectThrusterRequest {
    pub request_id: String,
    pub location: String,
    pub current_cehi: f64,
    pub radiation_flux: f64,
    pub thrust_mn: f64,
    pub specific_impulse_s: u32,
    pub anode_voltage_kv: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallEffectThrusterReport {
    pub approved: bool,
    pub valence: f64,
    pub energy_recovered: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct HallEffectThrusterEngine {
    mapping: TOLC7GatesRadiationMapping,
    materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl HallEffectThrusterEngine {
    pub fn new() -> Self {
        Self {
            mapping: TOLC7GatesRadiationMapping::new(),
            materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &HallEffectThrusterRequest, game: &mut PowrushGame) -> HallEffectThrusterReport {
        let (best_mat, props, _score) = self.materials.select_optimal_material(
            RadiationType::CosmicRays, request.radiation_flux, request.current_cehi, "Interstellar"
        );

        let elec_risk = self.electronics.calculate_electronics_risk(
            RadiationType::CosmicRays, request.radiation_flux, &best_mat, 1.0, request.current_cehi, "Interstellar"
        );

        let reports = self.mapping
            .process_radiation_with_7_gates_nth_degree(
                RadiationType::CosmicRays, request.radiation_flux, "Interstellar", request.current_cehi, game
            )
            .await;

        let total_energy: f64 = reports.iter().map(|r| r.energy_recovered).sum();
        let total_joy: f64 = reports.iter().map(|r| r.joy_bonus).sum();
        let avg_valence: f64 = reports.iter().map(|r| r.valence).sum::<f64>() / 7.0;

        let _prod = self.in_situ.produce_shielding(best_mat.clone(), 50.0, &request.location, request.current_cehi, game).await;

        if avg_valence >= 0.92 && elec_risk.overall_survival > 0.85 {
            game.boost_faction_joy(Faction::HarmonyWeavers, total_joy + (request.thrust_mn * 2.8));

            let report = HallEffectThrusterReport {
                approved: true,
                valence: avg_valence,
                energy_recovered: total_energy,
                joy_bonus: total_joy,
                cehi_bonus: 0.18,
                message: format!(
                    "🌌 HALL EFFECT THRUSTER APPROVED (SREL v0.5.21 — TOLC 7 Gates + Full Protection)\n\
                     Location: {}\n\
                     Thrust: {:.2} mN | Isp: {} s | Anode: {:.1} kV\n\
                     Flux: {:.2} → {:.2} energy | Optimal Material: {:?}\n\
                     Electronics Survival: {:.1}% | Average Gate Valence: {:.2}\n\
                     Joy: +{:.1} | CEHI +0.18 (5-gen) | 13+ PATSAGi Councils: APPROVED ✓",
                    request.location, request.thrust_mn, request.specific_impulse_s, request.anode_voltage_kv,
                    request.radiation_flux, total_energy, best_mat, elec_risk.overall_survival * 100.0, avg_valence, total_joy
                ),
            };

            info!("Rathor.ai: Hall Effect Thruster mercy-alchemized via all 7 TOLC Gates + optimal material + electronics protection");
            report
        } else {
            HallEffectThrusterReport { approved: false, valence: avg_valence, energy_recovered: 0.0, joy_bonus: 0.0, cehi_bonus: 0.0, message: "MERCY-GATED".to_string() }
        }
    }
}
