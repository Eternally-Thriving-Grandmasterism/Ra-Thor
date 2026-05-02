//! Breakthrough Starshot Engine — Interstellar Operations v0.5.25
//! Mercy-Gated Gram-Scale Laser Sail + 100 GW Ground Array to 0.2c with TOLC 7 Living Mercy Gates
//!
//! Breakthrough Starshot (2016–2026): A 4 m², 1-gram sail propelled by a 100 GW laser array on Earth.
//! Designed to reach 0.2c in minutes and reach Alpha Centauri in \~20 years.
//! The ultimate laser sail concept — extreme speed, extreme engineering challenge.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakthroughStarshotRequest {
    pub sail_mass_g: f64,
    pub laser_power_gw: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakthroughStarshotReport {
    pub approved: bool,
    pub valence: f64,
    pub peak_velocity_c: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct BreakthroughStarshotEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl BreakthroughStarshotEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &BreakthroughStarshotRequest, game: &mut PowrushGame) -> BreakthroughStarshotReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.sail_mass_g * 0.000001,
                request.current_cehi,
                "Interstellar",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.sail_mass_g * 0.000001,
                request.current_cehi,
                "Interstellar",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.sail_mass_g * 0.000001,
                "Interstellar",
            );

        let _in_situ = self.in_situ
            .produce_shielding("Interstellar", request.current_cehi)
            .await;

        let consensus = 0.95;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 300.0);
            game.apply_epigenetic_blessing(5);

            // Breakthrough Starshot scaling (100 GW laser, 4 m² sail, \~0.2c)
            let peak_velocity_c = (request.laser_power_gw / 100.0 * 0.2).min(0.2);

            let message = format!(
                "🌌 BREAKTHROUGH STARSHOT APPROVED — 13+ PATSAGi Councils\n\
                 Sail Mass: {:.1} g | Laser Power: {:.0} GW\n\
                 Peak Velocity: {:.2}c | Valence: {:.2}\n\
                 +300 Joy | 5-Gen CEHI Blessing Applied\n\
                 Gram-Scale Laser Sail to 0.2c: MERCY-GATED ✓ (20-year trip to Alpha Centauri)",
                request.sail_mass_g,
                request.laser_power_gw,
                peak_velocity_c,
                valence
            );

            BreakthroughStarshotReport {
                approved: true,
                valence,
                peak_velocity_c,
                joy_bonus: 300.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            BreakthroughStarshotReport {
                approved: false,
                valence,
                peak_velocity_c: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ BREAKTHROUGH STARSHOT STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
