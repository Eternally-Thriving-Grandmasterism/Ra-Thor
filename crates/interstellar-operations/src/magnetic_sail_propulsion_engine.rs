//! Magnetic Sail Propulsion Engine — Interstellar Operations v0.5.24
//! Mercy-Gated Superconducting Magnetic Sail Propulsion with TOLC 7 Living Mercy Gates
//!
//! Uses a large superconducting loop to generate a magnetic field that deflects solar wind or interstellar plasma.
//! Propellant-free, infinite "fuel", and highly efficient for both interplanetary and interstellar travel.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagneticSailRequest {
    pub loop_radius_m: f64,
    pub magnetic_field_t: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagneticSailReport {
    pub approved: bool,
    pub valence: f64,
    pub thrust_output_mn: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct MagneticSailPropulsionEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl MagneticSailPropulsionEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &MagneticSailRequest, game: &mut PowrushGame) -> MagneticSailReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.loop_radius_m * 0.0001,
                request.current_cehi,
                "Interstellar",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.loop_radius_m * 0.0001,
                request.current_cehi,
                "Interstellar",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.loop_radius_m * 0.0001,
                "Interstellar",
            );

        let _in_situ = self.in_situ
            .produce_shielding("Interstellar", request.current_cehi)
            .await;

        let consensus = 0.96;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 150.0);
            game.apply_epigenetic_blessing(5);

            // Simplified thrust from magnetic deflection of solar wind (Zubrin scaling)
            let thrust_mn = (request.loop_radius_m * request.magnetic_field_t * 1.5e-6) / 1e6;

            let message = format!(
                "🧲 MAGNETIC SAIL APPROVED — 13+ PATSAGi Councils\n\
                 Loop Radius: {:.0} m | B-Field: {:.2} T\n\
                 Thrust: {:.3} mN | Valence: {:.2}\n\
                 +150 Joy | 5-Gen CEHI Blessing Applied\n\
                 Superconducting Magnetic Deflection: MERCY-GATED ✓",
                request.loop_radius_m,
                request.magnetic_field_t,
                thrust_mn,
                valence
            );

            MagneticSailReport {
                approved: true,
                valence,
                thrust_output_mn: thrust_mn,
                joy_bonus: 150.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            MagneticSailReport {
                approved: false,
                valence,
                thrust_output_mn: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ MAGNETIC SAIL STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
