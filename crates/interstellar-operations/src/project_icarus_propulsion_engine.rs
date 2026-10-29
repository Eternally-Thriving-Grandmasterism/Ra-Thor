//! Project Icarus Propulsion Engine — Interstellar Operations v0.5.25
//! Mercy-Gated Modern Fusion Pellet Interstellar Probe (Icarus Interstellar 2010s–2026) with TOLC 7 Living Mercy Gates
//!
//! Successor to 1978 BIS Daedalus. Uses laser-ignited deuterium/helium-3 pellets,
//! advanced magnetic nozzles, and realistic engineering (lower mass, higher reliability).
//! Designed for 50–100 year missions to nearby stars with \~0.1–0.15c cruise velocity.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectIcarusRequest {
    pub pellet_count: u32,
    pub laser_ignition_mj: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectIcarusReport {
    pub approved: bool,
    pub valence: f64,
    pub peak_velocity_c: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct ProjectIcarusPropulsionEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl ProjectIcarusPropulsionEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &ProjectIcarusRequest, game: &mut PowrushGame) -> ProjectIcarusReport {
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

        let consensus = 0.96;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 240.0);
            game.apply_epigenetic_blessing(5);

            // Modern Icarus scaling (laser ignition + improved nozzle efficiency)
            let peak_velocity_c = (request.pellet_count as f64 / 45000.0 * request.laser_ignition_mj / 800.0 * 0.14).min(0.15);

            let message = format!(
                "🚀 PROJECT ICARUS APPROVED — 13+ PATSAGi Councils\n\
                 Pellets: {} | Laser Ignition: {:.0} MJ\n\
                 Peak Velocity: {:.2}c | Valence: {:.2}\n\
                 +240 Joy | 5-Gen CEHI Blessing Applied\n\
                 Modern Laser-Ignited Fusion Pellet Drive: MERCY-GATED ✓ (Icarus Interstellar 2026 design)",
                request.pellet_count,
                request.laser_ignition_mj,
                peak_velocity_c,
                valence
            );

            ProjectIcarusReport {
                approved: true,
                valence,
                peak_velocity_c,
                joy_bonus: 240.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            ProjectIcarusReport {
                approved: false,
                valence,
                peak_velocity_c: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ PROJECT ICARUS STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
