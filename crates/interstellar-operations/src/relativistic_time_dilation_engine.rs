//! Relativistic Time Dilation Engine — Interstellar Operations v0.5.25
//! Mercy-Gated Special Relativity Calculator for High-Speed Interstellar Missions with TOLC 7 Living Mercy Gates
//!
//! Calculates proper time (ship time) vs Earth time, length contraction, and mission planning
//! for velocities up to 0.2c+ (Starshot, Daedalus, Icarus, etc.). Essential for crewed missions.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativisticTimeDilationRequest {
    pub velocity_c: f64,
    pub earth_time_years: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativisticTimeDilationReport {
    pub approved: bool,
    pub valence: f64,
    pub proper_time_years: f64,
    pub length_contraction_factor: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct RelativisticTimeDilationEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl RelativisticTimeDilationEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &RelativisticTimeDilationRequest, game: &mut PowrushGame) -> RelativisticTimeDilationReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.velocity_c * 10.0,
                request.current_cehi,
                "DeepSpace",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.velocity_c * 10.0,
                request.current_cehi,
                "DeepSpace",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.velocity_c * 10.0,
                "DeepSpace",
            );

        let _in_situ = self.in_situ
            .produce_shielding("DeepSpace", request.current_cehi)
            .await;

        let consensus = 0.96;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 260.0);
            game.apply_epigenetic_blessing(5);

            // Special relativity formulas (gamma = 1 / sqrt(1 - v²/c²))
            let gamma = 1.0 / (1.0 - request.velocity_c.powi(2)).sqrt();
            let proper_time_years = request.earth_time_years / gamma;
            let length_contraction_factor = 1.0 / gamma;

            let message = format!(
                "⏳ RELATIVISTIC TIME DILATION APPROVED — 13+ PATSAGi Councils\n\
                 Velocity: {:.2}c | Earth Time: {:.1} years\n\
                 Proper Time (ship): {:.2} years | Gamma: {:.2}\n\
                 +260 Joy | 5-Gen CEHI Blessing Applied\n\
                 Time Dilation + Length Contraction: MERCY-GATED ✓",
                request.velocity_c,
                request.earth_time_years,
                proper_time_years,
                gamma
            );

            RelativisticTimeDilationReport {
                approved: true,
                valence,
                proper_time_years,
                length_contraction_factor,
                joy_bonus: 260.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            RelativisticTimeDilationReport {
                approved: false,
                valence,
                proper_time_years: 0.0,
                length_contraction_factor: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ RELATIVISTIC TIME DILATION STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
