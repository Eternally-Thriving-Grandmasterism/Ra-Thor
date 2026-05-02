//! Destiny Ship Seed Engine — Interstellar Operations v0.5.21
//! Generational Seed-Ship Logic with TOLC 7 Living Mercy Gates

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DestinyShipSeedRequest {
    pub crew_size: u32,
    pub generations: u8,
    pub seed_ship_class: String,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DestinyShipSeedReport {
    pub approved: bool,
    pub valence: f64,
    pub generational_stability: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct DestinyShipSeedEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl DestinyShipSeedEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    pub async fn evaluate(&self, request: &DestinyShipSeedRequest, game: &mut PowrushGame) -> DestinyShipSeedReport {
        let valence = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::CosmicRays,
                0.05,
                request.current_cehi,
                "DeepSpace",
            )
            .await
            .avg_valence;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::CosmicRays,
                0.05,
                request.current_cehi,
                "DeepSpace",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::CosmicRays,
                0.05,
                "DeepSpace",
            );

        let _in_situ = self.in_situ
            .produce_shielding("DeepSpace", request.current_cehi)
            .await;

        let consensus = 0.96;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 150.0);
            game.apply_epigenetic_blessing(5);

            let message = format!(
                "🌱 DESTINY SHIP SEED ENGINE APPROVED — 13+ PATSAGi Councils\n\
                 Class: {} | Crew: {} | Generations: {}\n\
                 Valence: {:.2} | Survival: {:.2}\n\
                 +150 Joy | 5-Gen CEHI Blessing (Eternal Legacy Locked)\n\
                 Generational Seed-Ship: MERCY-GATED ✓",
                request.seed_ship_class,
                request.crew_size,
                request.generations,
                valence,
                elec_risk.overall_survival
            );

            DestinyShipSeedReport {
                approved: true,
                valence,
                generational_stability: 0.99,
                joy_bonus: 150.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            DestinyShipSeedReport {
                approved: false,
                valence,
                generational_stability: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ DESTINY SHIP SEED STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
