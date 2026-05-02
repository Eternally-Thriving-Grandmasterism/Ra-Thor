//! Stargate Dialing Computer — Interstellar Operations v0.5.21
//! Symbolic + Quantum Address Dialing with TOLC 7 Living Mercy Gates

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StargateDialingRequest {
    pub address_symbols: String,
    pub chevrons_locked: u8,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StargateDialingReport {
    pub approved: bool,
    pub valence: f64,
    pub connection_stability: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct StargateDialingComputer {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl StargateDialingComputer {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &StargateDialingRequest, game: &mut PowrushGame) -> StargateDialingReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::SolarFlare,
                request.chevrons_locked as f64 * 0.05,
                request.current_cehi,
                "Interstellar",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::SolarFlare,
                request.chevrons_locked as f64 * 0.05,
                request.current_cehi,
                "Interstellar",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::SolarFlare,
                request.chevrons_locked as f64 * 0.05,
                "Interstellar",
            );

        let _in_situ = self.in_situ
            .produce_shielding("Interstellar", request.current_cehi)
            .await;

        let consensus = 0.95;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 120.0);
            game.apply_epigenetic_blessing(5);

            let message = format!(
                "🌀 STARGATE DIALING COMPUTER APPROVED — 13+ PATSAGi Councils\n\
                 Address: {} | Chevrons: {}/9\n\
                 Valence: {:.2} | Survival: {:.2}\n\
                 +120 Joy | 5-Gen CEHI Blessing Applied\n\
                 Wormhole Connection: MERCY-GATED ✓",
                request.address_symbols,
                request.chevrons_locked,
                valence,
                elec_risk.overall_survival
            );

            StargateDialingReport {
                approved: true,
                valence,
                connection_stability: 0.98,
                joy_bonus: 120.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            StargateDialingReport {
                approved: false,
                valence,
                connection_stability: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ STARGATE DIALING STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
