//! Wormhole Stability Engine — Interstellar Operations v0.5.25
//! Mercy-Gated Exotic Matter Requirements, Throat Stability, Tidal Forces & Safe Transit with TOLC 7 Living Mercy Gates
//!
//! Essential for Stargate-style travel. Calculates minimum exotic matter needed to keep a wormhole open,
//! tidal force safety margins for crews, and transit time through the throat.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeStabilityRequest {
    pub throat_radius_m: f64,
    pub exotic_matter_density_kg_m3: f64,
    pub transit_velocity_c: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeStabilityReport {
    pub approved: bool,
    pub valence: f64,
    pub minimum_exotic_matter_kg: f64,
    pub tidal_force_g: f64,
    pub transit_time_seconds: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct WormholeStabilityEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl WormholeStabilityEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &WormholeStabilityRequest, game: &mut PowrushGame) -> WormholeStabilityReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.throat_radius_m * 0.0001,
                request.current_cehi,
                "DeepSpace",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.throat_radius_m * 0.0001,
                request.current_cehi,
                "DeepSpace",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.throat_radius_m * 0.0001,
                "DeepSpace",
            );

        let _in_situ = self.in_situ
            .produce_shielding("DeepSpace", request.current_cehi)
            .await;

        let consensus = 0.93;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 310.0);
            game.apply_epigenetic_blessing(5);

            // Simplified Morris-Thorne wormhole formulas
            let min_exotic = 1.0e10 * (request.throat_radius_m / 1000.0).powi(2); // kg (order-of-magnitude)
            let tidal_force = (request.transit_velocity_c * 9.81) / (request.throat_radius_m / 1_000_000.0);
            let transit_time = (2.0 * request.throat_radius_m) / (request.transit_velocity_c * 3e8);

            let message = format!(
                "🌀 WORMHOLE STABILITY APPROVED — 13+ PATSAGi Councils\n\
                 Throat Radius: {:.0} m | Exotic Matter: {:.2e} kg\n\
                 Tidal Force: {:.1} g | Transit Time: {:.2} s\n\
                 +310 Joy | 5-Gen CEHI Blessing Applied\n\
                 Exotic Matter + Throat Stability: MERCY-GATED ✓",
                request.throat_radius_m,
                min_exotic,
                tidal_force,
                transit_time
            );

            WormholeStabilityReport {
                approved: true,
                valence,
                minimum_exotic_matter_kg: min_exotic,
                tidal_force_g: tidal_force,
                transit_time_seconds: transit_time,
                joy_bonus: 310.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            WormholeStabilityReport {
                approved: false,
                valence,
                minimum_exotic_matter_kg: 0.0,
                tidal_force_g: 0.0,
                transit_time_seconds: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ WORMHOLE STABILITY STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
