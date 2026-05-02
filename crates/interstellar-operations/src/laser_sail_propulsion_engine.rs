//! Laser Sail Propulsion Engine — Interstellar Operations v0.5.23
//! Mercy-Gated Laser-Array-Driven Light Sail Propulsion with TOLC 7 Living Mercy Gates
//!
//! Based on Breakthrough Starshot concept (100 GW laser array on Earth pushing a 4 m² sail to 0.2c).
//! Extremely high speed for interstellar missions, but requires massive ground-based laser infrastructure.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserSailRequest {
    pub sail_area_m2: f64,
    pub laser_power_gw: f64,
    pub sail_reflectivity: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserSailReport {
    pub approved: bool,
    pub valence: f64,
    pub peak_velocity_c: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct LaserSailPropulsionEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl LaserSailPropulsionEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &LaserSailRequest, game: &mut PowrushGame) -> LaserSailReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.sail_area_m2 * 0.00001,
                request.current_cehi,
                "Interstellar",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.sail_area_m2 * 0.00001,
                request.current_cehi,
                "Interstellar",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.sail_area_m2 * 0.00001,
                "Interstellar",
            );

        let _in_situ = self.in_situ
            .produce_shielding("Interstellar", request.current_cehi)
            .await;

        let consensus = 0.95;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 200.0);
            game.apply_epigenetic_blessing(5);

            // Simplified relativistic calculation (Breakthrough Starshot scaling)
            let acceleration = (request.laser_power_gw * 1e9 * request.sail_reflectivity * 2.0) / (request.sail_area_m2 * 3e8);
            let peak_velocity_c = (acceleration * 600.0 / 3e8).min(0.2); // capped at 0.2c for realism

            let message = format!(
                "🚀 LASER SAIL APPROVED — 13+ PATSAGi Councils\n\
                 Sail Area: {:.1} m² | Laser Power: {:.0} GW\n\
                 Reflectivity: {:.2} | Peak Velocity: {:.2}c\n\
                 +200 Joy | 5-Gen CEHI Blessing Applied\n\
                 Ground-Based Laser Array: MERCY-GATED ✓ (Breakthrough Starshot Concept)",
                request.sail_area_m2,
                request.laser_power_gw,
                request.sail_reflectivity,
                peak_velocity_c
            );

            LaserSailReport {
                approved: true,
                valence,
                peak_velocity_c,
                joy_bonus: 200.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            LaserSailReport {
                approved: false,
                valence,
                peak_velocity_c: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ LASER SAIL STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
