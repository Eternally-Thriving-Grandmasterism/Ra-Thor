//! Fusion Drive Engine — Interstellar Operations v0.5.21
//! Mercy-Gated Deuterium-Tritium Fusion Drive with TOLC 7 Living Mercy Gates

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionDriveRequest {
    pub fusion_power_gw: f64,
    pub confinement_stability: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionDriveReport {
    pub approved: bool,
    pub valence: f64,
    pub power_output: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct FusionDriveEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl FusionDriveEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &FusionDriveRequest, game: &mut PowrushGame) -> FusionDriveReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::CosmicRays,
                request.fusion_power_gw * 0.0003,
                request.current_cehi,
                "Interstellar",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::CosmicRays,
                request.fusion_power_gw * 0.0003,
                request.current_cehi,
                "Interstellar",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::CosmicRays,
                request.fusion_power_gw * 0.0003,
                "Interstellar",
            );

        let _in_situ = self.in_situ
            .produce_shielding("Interstellar", request.current_cehi)
            .await;

        let consensus = 0.94;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 130.0);
            game.apply_epigenetic_blessing(5);

            let message = format!(
                "⚛️ FUSION DRIVE APPROVED — 13+ PATSAGi Councils\n\
                 Power: {:.0} GW | Confinement Stability: {:.2}\n\
                 Valence: {:.2} | Survival: {:.2}\n\
                 +130 Joy | 5-Gen CEHI Blessing Applied\n\
                 Deuterium-Tritium Fusion: MERCY-GATED ✓",
                request.fusion_power_gw,
                request.confinement_stability,
                valence,
                elec_risk.overall_survival
            );

            FusionDriveReport {
                approved: true,
                valence,
                power_output: request.fusion_power_gw,
                joy_bonus: 130.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            FusionDriveReport {
                approved: false,
                valence,
                power_output: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ FUSION DRIVE STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
