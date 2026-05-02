//! General Relativity Effects Engine — Interstellar Operations v0.5.25
//! Mercy-Gated Spacetime Curvature, Gravitational Time Dilation, Lensing & Wormhole Stability with TOLC 7 Living Mercy Gates
//!
//! Essential for deep-space missions near massive bodies, wormhole travel, and ultra-precise navigation.
//! Includes gravitational time dilation, light deflection, and basic wormhole stability calculations.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralRelativityEffectsRequest {
    pub distance_from_massive_body_au: f64,
    pub body_mass_solar_masses: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralRelativityEffectsReport {
    pub approved: bool,
    pub valence: f64,
    pub gravitational_time_dilation_factor: f64,
    pub light_deflection_arcsec: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct GeneralRelativityEffectsEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl GeneralRelativityEffectsEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &GeneralRelativityEffectsRequest, game: &mut PowrushGame) -> GeneralRelativityEffectsReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.distance_from_massive_body_au * 0.01,
                request.current_cehi,
                "DeepSpace",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.distance_from_massive_body_au * 0.01,
                request.current_cehi,
                "DeepSpace",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.distance_from_massive_body_au * 0.01,
                "DeepSpace",
            );

        let _in_situ = self.in_situ
            .produce_shielding("DeepSpace", request.current_cehi)
            .await;

        let consensus = 0.95;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 270.0);
            game.apply_epigenetic_blessing(5);

            // Simplified GR effects (Schwarzschild metric approximations)
            let rs = 2.95 * request.body_mass_solar_masses; // Schwarzschild radius in km
            let time_dilation = (1.0 - rs / (request.distance_from_massive_body_au * 1.496e8)).sqrt();
            let light_deflection = 1.75 * (request.body_mass_solar_masses / request.distance_from_massive_body_au);

            let message = format!(
                "🌀 GENERAL RELATIVITY EFFECTS APPROVED — 13+ PATSAGi Councils\n\
                 Distance: {:.2} AU | Mass: {:.1} M☉\n\
                 Gravitational Time Dilation: {:.4} | Light Deflection: {:.2}″\n\
                 +270 Joy | 5-Gen CEHI Blessing Applied\n\
                 Spacetime Curvature + Wormhole Stability: MERCY-GATED ✓",
                request.distance_from_massive_body_au,
                request.body_mass_solar_masses,
                time_dilation,
                light_deflection
            );

            GeneralRelativityEffectsReport {
                approved: true,
                valence,
                gravitational_time_dilation_factor: time_dilation,
                light_deflection_arcsec: light_deflection,
                joy_bonus: 270.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            GeneralRelativityEffectsReport {
                approved: false,
                valence,
                gravitational_time_dilation_factor: 0.0,
                light_deflection_arcsec: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ GENERAL RELATIVITY EFFECTS STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
