//! Warp Core Engine — Interstellar Operations v0.5.21
//! Mercy-Gated Matter-Antimatter Annihilation with TOLC 7 Living Mercy Gates

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpCoreRequest {
    pub stability_threshold: f64,
    pub power_output_tw: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpCoreReport {
    pub approved: bool,
    pub valence: f64,
    pub energy_released: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct WarpCoreEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl WarpCoreEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &WarpCoreRequest, game: &mut PowrushGame) -> WarpCoreReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::CosmicRays,
                request.power_output_tw * 0.0001,
                request.current_cehi,
                "Interstellar",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::CosmicRays,
                request.power_output_tw * 0.0001,
                request.current_cehi,
                "Interstellar",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::CosmicRays,
                request.power_output_tw * 0.0001,
                "Interstellar",
            );

        let _in_situ = self.in_situ
            .produce_shielding("Interstellar", request.current_cehi)
            .await;

        let consensus = 0.93; // Quantum swarm consensus

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 95.0);
            game.apply_epigenetic_blessing(5);

            let message = format!(
                "🌌 WARP CORE APPROVED — 13+ PATSAGi Councils\n\
                 Stability: {:.2} | Power: {:.0} TW\n\
                 Valence: {:.2} | Survival: {:.2}\n\
                 +95 Joy | 5-Gen CEHI Blessing Applied\n\
                 Matter-Antimatter Annihilation: MERCY-GATED ✓",
                request.stability_threshold,
                request.power_output_tw,
                valence,
                elec_risk.overall_survival
            );

            WarpCoreReport {
                approved: true,
                valence,
                energy_released: request.power_output_tw,
                joy_bonus: 95.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            WarpCoreReport {
                approved: false,
                valence,
                energy_released: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ WARP CORE STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
