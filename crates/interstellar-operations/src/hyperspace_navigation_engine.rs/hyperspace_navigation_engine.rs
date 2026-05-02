//! Hyperspace Navigation Engine — Interstellar Operations v0.5.21
//! Mercy-Gated Hyperspace Jump Navigation with TOLC 7 Living Mercy Gates

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperspaceNavigationRequest {
    pub destination: String,
    pub jump_distance_ly: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperspaceNavigationReport {
    pub approved: bool,
    pub valence: f64,
    pub navigation_accuracy: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct HyperspaceNavigationEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl HyperspaceNavigationEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &HyperspaceNavigationRequest, game: &mut PowrushGame) -> HyperspaceNavigationReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::CosmicRays,
                request.jump_distance_ly * 0.001,
                request.current_cehi,
                "Interstellar",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::CosmicRays,
                request.jump_distance_ly * 0.001,
                request.current_cehi,
                "Interstellar",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::CosmicRays,
                request.jump_distance_ly * 0.001,
                "Interstellar",
            );

        let _in_situ = self.in_situ
            .produce_shielding("Interstellar", request.current_cehi)
            .await;

        let consensus = 0.92;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 85.0);
            game.apply_epigenetic_blessing(5);

            let message = format!(
                "🌌 HYPERSPACE NAVIGATION APPROVED — 13+ PATSAGi Councils\n\
                 Destination: {} | Distance: {:.1} ly\n\
                 Valence: {:.2} | Survival: {:.2}\n\
                 +85 Joy | 5-Gen CEHI Blessing Applied\n\
                 Hyperspace Jump: MERCY-GATED ✓",
                request.destination,
                request.jump_distance_ly,
                valence,
                elec_risk.overall_survival
            );

            HyperspaceNavigationReport {
                approved: true,
                valence,
                navigation_accuracy: 0.97,
                joy_bonus: 85.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            HyperspaceNavigationReport {
                approved: false,
                valence,
                navigation_accuracy: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ HYPERSPACE NAVIGATION STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
