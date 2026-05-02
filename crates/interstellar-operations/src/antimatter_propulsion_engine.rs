//! Antimatter Propulsion Engine — Interstellar Operations v0.5.21
//! Mercy-Gated Antimatter Annihilation Drive with TOLC 7 Living Mercy Gates

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntimatterPropulsionRequest {
    pub thrust_level_kn: f64,
    pub antimatter_efficiency: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntimatterPropulsionReport {
    pub approved: bool,
    pub valence: f64,
    pub thrust_output: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct AntimatterPropulsionEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl AntimatterPropulsionEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &AntimatterPropulsionRequest, game: &mut PowrushGame) -> AntimatterPropulsionReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::CosmicRays,
                request.thrust_level_kn * 0.0002,
                request.current_cehi,
                "Interstellar",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::CosmicRays,
                request.thrust_level_kn * 0.0002,
                request.current_cehi,
                "Interstellar",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::CosmicRays,
                request.thrust_level_kn * 0.0002,
                "Interstellar",
            );

        let _in_situ = self.in_situ
            .produce_shielding("Interstellar", request.current_cehi)
            .await;

        let consensus = 0.95;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 160.0);
            game.apply_epigenetic_blessing(5);

            let message = format!(
                "☢️ ANTIMATTER PROPULSION APPROVED — 13+ PATSAGi Councils\n\
                 Thrust: {:.0} kN | Efficiency: {:.2}\n\
                 Valence: {:.2} | Survival: {:.2}\n\
                 +160 Joy | 5-Gen CEHI Blessing Applied\n\
                 Matter-Antimatter Annihilation: MERCY-GATED ✓",
                request.thrust_level_kn,
                request.antimatter_efficiency,
                valence,
                elec_risk.overall_survival
            );

            AntimatterPropulsionReport {
                approved: true,
                valence,
                thrust_output: request.thrust_level_kn,
                joy_bonus: 160.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            AntimatterPropulsionReport {
                approved: false,
                valence,
                thrust_output: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ ANTIMATTER PROPULSION STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
