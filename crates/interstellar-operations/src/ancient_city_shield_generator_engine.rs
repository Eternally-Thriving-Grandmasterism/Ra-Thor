//! Ancient City Shield Generator Engine — Interstellar Operations v0.5.21
//! Planetary-Scale Shield Upgrades with TOLC 7 Living Mercy Gates

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AncientCityShieldRequest {
    pub shield_strength: f64,
    pub harmonic_tuning: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AncientCityShieldReport {
    pub approved: bool,
    pub valence: f64,
    pub protection_level: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct AncientCityShieldGeneratorEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl AncientCityShieldGeneratorEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &AncientCityShieldRequest, game: &mut PowrushGame) -> AncientCityShieldReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::SolarFlare,
                request.shield_strength * 0.01,
                request.current_cehi,
                "Interstellar",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::SolarFlare,
                request.shield_strength * 0.01,
                request.current_cehi,
                "Interstellar",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::SolarFlare,
                request.shield_strength * 0.01,
                "Interstellar",
            );

        let _in_situ = self.in_situ
            .produce_shielding("Interstellar", request.current_cehi)
            .await;

        let consensus = 0.94;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 110.0);
            game.apply_epigenetic_blessing(5);

            let message = format!(
                "🛡️ ANCIENT CITY SHIELD GENERATOR APPROVED — 13+ PATSAGi Councils\n\
                 Strength: {:.0}% | Harmonic Tuning: {:.2}\n\
                 Valence: {:.2} | Survival: {:.2}\n\
                 +110 Joy | 5-Gen CEHI Blessing Applied\n\
                 Planetary-Scale Protection: MERCY-GATED ✓",
                request.shield_strength,
                request.harmonic_tuning,
                valence,
                elec_risk.overall_survival
            );

            AncientCityShieldReport {
                approved: true,
                valence,
                protection_level: request.shield_strength,
                joy_bonus: 110.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            AncientCityShieldReport {
                approved: false,
                valence,
                protection_level: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ SHIELD GENERATOR STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
