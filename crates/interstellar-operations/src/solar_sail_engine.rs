//! Solar Sail Engine — Interstellar Operations v0.5.23
//! Mercy-Gated Photon-Powered Solar Sail Propulsion with TOLC 7 Living Mercy Gates
//!
//! Uses solar radiation pressure (photons) for propellant-free acceleration.
//! Ideal for deep-space, long-duration missions with infinite "fuel" from the Sun.
//! Extremely low thrust but extremely high efficiency and mercy alignment.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarSailRequest {
    pub sail_area_m2: f64,
    pub distance_from_sun_au: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarSailReport {
    pub approved: bool,
    pub valence: f64,
    pub thrust_output_mn: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct SolarSailEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl SolarSailEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &SolarSailRequest, game: &mut PowrushGame) -> SolarSailReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::SolarFlare,
                request.sail_area_m2 * 0.000001,
                request.current_cehi,
                "Interstellar",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::SolarFlare,
                request.sail_area_m2 * 0.000001,
                request.current_cehi,
                "Interstellar",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::SolarFlare,
                request.sail_area_m2 * 0.000001,
                "Interstellar",
            );

        let _in_situ = self.in_situ
            .produce_shielding("Interstellar", request.current_cehi)
            .await;

        let consensus = 0.96;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 120.0);
            game.apply_epigenetic_blessing(5);

            // Approximate thrust using solar radiation pressure (9.08 μN/m² at 1 AU)
            let thrust_mn = (request.sail_area_m2 * 9.08e-6) / (request.distance_from_sun_au * request.distance_from_sun_au);

            let message = format!(
                "🌞 SOLAR SAIL APPROVED — 13+ PATSAGi Councils\n\
                 Sail Area: {:.0} m² | Distance: {:.2} AU\n\
                 Thrust: {:.3} mN | Valence: {:.2}\n\
                 +120 Joy | 5-Gen CEHI Blessing Applied\n\
                 Photon-Powered, Propellant-Free: MERCY-GATED ✓",
                request.sail_area_m2,
                request.distance_from_sun_au,
                thrust_mn,
                valence
            );

            SolarSailReport {
                approved: true,
                valence,
                thrust_output_mn: thrust_mn,
                joy_bonus: 120.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            SolarSailReport {
                approved: false,
                valence,
                thrust_output_mn: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ SOLAR SAIL STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
