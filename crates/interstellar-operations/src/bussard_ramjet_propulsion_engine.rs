//! Bussard Ramjet Propulsion Engine — Interstellar Operations v0.5.25
//! Mercy-Gated Interstellar Hydrogen Scoop + Fusion Ramjet with TOLC 7 Living Mercy Gates
//!
//! Classic Bussard 1960 concept: a huge magnetic scoop collects interstellar hydrogen,
//! compresses it, fuses it for thrust, and expels the exhaust at relativistic speeds.
//! Theoretically capable of reaching 0.1–0.2c over decades with no onboard propellant.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BussardRamjetRequest {
    pub scoop_radius_m: f64,
    pub fusion_efficiency: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BussardRamjetReport {
    pub approved: bool,
    pub valence: f64,
    pub peak_velocity_c: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct BussardRamjetPropulsionEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl BussardRamjetPropulsionEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &BussardRamjetRequest, game: &mut PowrushGame) -> BussardRamjetReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.scoop_radius_m * 0.00001,
                request.current_cehi,
                "DeepSpace",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.scoop_radius_m * 0.00001,
                request.current_cehi,
                "DeepSpace",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.scoop_radius_m * 0.00001,
                "DeepSpace",
            );

        let _in_situ = self.in_situ
            .produce_shielding("DeepSpace", request.current_cehi)
            .await;

        let consensus = 0.94;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 180.0);
            game.apply_epigenetic_blessing(5);

            // Simplified Bussard scaling (thrust from collected H + fusion)
            let peak_velocity_c = (request.scoop_radius_m / 1_000_000.0 * request.fusion_efficiency * 0.15).min(0.18);

            let message = format!(
                "🌌 BUSSARD RAMJET APPROVED — 13+ PATSAGi Councils\n\
                 Scoop Radius: {:.0} km | Efficiency: {:.2}\n\
                 Peak Velocity: {:.2}c | Valence: {:.2}\n\
                 +180 Joy | 5-Gen CEHI Blessing Applied\n\
                 Interstellar Hydrogen Scoop + Fusion: MERCY-GATED ✓",
                request.scoop_radius_m / 1000.0,
                request.fusion_efficiency,
                peak_velocity_c,
                valence
            );

            BussardRamjetReport {
                approved: true,
                valence,
                peak_velocity_c,
                joy_bonus: 180.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            BussardRamjetReport {
                approved: false,
                valence,
                peak_velocity_c: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ BUSSARD RAMJET STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
