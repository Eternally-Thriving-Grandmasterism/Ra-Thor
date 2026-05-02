//! Project Daedalus Propulsion Engine — Interstellar Operations v0.5.25
//! Mercy-Gated 1978 BIS Fusion Pellet Interstellar Probe with TOLC 7 Living Mercy Gates
//!
//! Classic British Interplanetary Society (BIS) Daedalus study (1973–1978):
//! A two-stage fusion rocket using 50,000+ deuterium/helium-3 pellets.
//! Magnetic nozzles accelerate the fusion exhaust to \~10,000 km/s.
//! Designed for a 50-year one-way trip to Barnard’s Star (5.9 ly) at 12% c.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectDaedalusRequest {
    pub pellet_count: u32,
    pub fusion_yield_mj: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectDaedalusReport {
    pub approved: bool,
    pub valence: f64,
    pub peak_velocity_c: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct ProjectDaedalusPropulsionEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl ProjectDaedalusPropulsionEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &ProjectDaedalusRequest, game: &mut PowrushGame) -> ProjectDaedalusReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.pellet_count as f64 * 0.000001,
                request.current_cehi,
                "DeepSpace",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.pellet_count as f64 * 0.000001,
                request.current_cehi,
                "DeepSpace",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.pellet_count as f64 * 0.000001,
                "DeepSpace",
            );

        let _in_situ = self.in_situ
            .produce_shielding("DeepSpace", request.current_cehi)
            .await;

        let consensus = 0.95;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 220.0);
            game.apply_epigenetic_blessing(5);

            // Classic Daedalus scaling (50,000 pellets, \~12% c final velocity)
            let peak_velocity_c = (request.pellet_count as f64 / 50000.0 * request.fusion_yield_mj / 1000.0 * 0.12).min(0.12);

            let message = format!(
                "🚀 PROJECT DAEDALUS APPROVED — 13+ PATSAGi Councils\n\
                 Pellets: {} | Yield: {:.0} MJ/pellet\n\
                 Peak Velocity: {:.2}c | Valence: {:.2}\n\
                 +220 Joy | 5-Gen CEHI Blessing Applied\n\
                 1978 BIS Fusion Pellet Ramjet: MERCY-GATED ✓ (50-year trip to Barnard’s Star)",
                request.pellet_count,
                request.fusion_yield_mj,
                peak_velocity_c,
                valence
            );

            ProjectDaedalusReport {
                approved: true,
                valence,
                peak_velocity_c,
                joy_bonus: 220.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            ProjectDaedalusReport {
                approved: false,
                valence,
                peak_velocity_c: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ PROJECT DAEDALUS STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
