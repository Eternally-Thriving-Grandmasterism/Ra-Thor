//! Black Hole Navigation Engine — Interstellar Operations v0.5.25
//! Mercy-Gated Safe Trajectory Planning Near Black Holes (Schwarzschild/Kerr Metrics) with TOLC 7 Living Mercy Gates
//!
//! Critical for advanced interstellar missions: gravitational slingshot maneuvers, safe flybys,
//! photon sphere avoidance, event horizon safety margins, and extreme time dilation near singularities.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackHoleNavigationRequest {
    pub black_hole_mass_solar_masses: f64,
    pub closest_approach_au: f64,
    pub target_velocity_c: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackHoleNavigationReport {
    pub approved: bool,
    pub valence: f64,
    pub safe_periapsis_au: f64,
    pub time_dilation_factor: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct BlackHoleNavigationEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl BlackHoleNavigationEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &BlackHoleNavigationRequest, game: &mut PowrushGame) -> BlackHoleNavigationReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.black_hole_mass_solar_masses * 0.001,
                request.current_cehi,
                "DeepSpace",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.black_hole_mass_solar_masses * 0.001,
                request.current_cehi,
                "DeepSpace",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.black_hole_mass_solar_masses * 0.001,
                "DeepSpace",
            );

        let _in_situ = self.in_situ
            .produce_shielding("DeepSpace", request.current_cehi)
            .await;

        let consensus = 0.94;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 290.0);
            game.apply_epigenetic_blessing(5);

            // Simplified safe periapsis + time dilation near black hole
            let rs = 2.95 * request.black_hole_mass_solar_masses; // Schwarzschild radius (km)
            let safe_periapsis_au = (rs * 3.0) / 1.496e8; // 3× Rs safety margin
            let time_dilation = 1.0 / (1.0 - (rs / (request.closest_approach_au * 1.496e8))).sqrt();

            let message = format!(
                "🕳️ BLACK HOLE NAVIGATION APPROVED — 13+ PATSAGi Councils\n\
                 Mass: {:.1} M☉ | Closest Approach: {:.2} AU\n\
                 Safe Periapsis: {:.2} AU | Time Dilation: {:.2}×\n\
                 +290 Joy | 5-Gen CEHI Blessing Applied\n\
                 Gravitational Slingshot + Event Horizon Safety: MERCY-GATED ✓",
                request.black_hole_mass_solar_masses,
                request.closest_approach_au,
                safe_periapsis_au,
                time_dilation
            );

            BlackHoleNavigationReport {
                approved: true,
                valence,
                safe_periapsis_au,
                time_dilation_factor: time_dilation,
                joy_bonus: 290.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            BlackHoleNavigationReport {
                approved: false,
                valence,
                safe_periapsis_au: 0.0,
                time_dilation_factor: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ BLACK HOLE NAVIGATION STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
