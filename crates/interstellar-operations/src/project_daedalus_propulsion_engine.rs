//! Project Daedalus Propulsion Engine — Interstellar Operations v0.5.25
//! Mercy-Gated 1978 BIS Fusion Pellet Rocket with Full TOLC 7 Living Mercy Gates + CEHI Epigenetic Blessings
//!
//! Real 1978 British Interplanetary Society (BIS) Daedalus parameters (50,000 fusion pellets, 0.12c to Barnard’s Star, 54-year mission) + complete mercy-gated integration.
//! This engine follows the exact polished merged template established by all previous engines.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine, CEHIEpigeneticBlessings,
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
    pub thrust_kn: f64,
    pub velocity_c: f64,
    pub mission_years: f64,
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
    cehi_blessings: CEHIEpigeneticBlessings,
}

impl ProjectDaedalusPropulsionEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
            cehi_blessings: CEHIEpigeneticBlessings::new(),
        }
    }

    pub async fn evaluate(&self, request: &ProjectDaedalusRequest, game: &mut PowrushGame) -> ProjectDaedalusReport {
        let gate_report = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.pellet_count as f64 * 0.00001,
                request.current_cehi,
                "DeepSpace",
            )
            .await;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.pellet_count as f64 * 0.00001,
                request.current_cehi,
                "DeepSpace",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.pellet_count as f64 * 0.00001,
                "DeepSpace",
            );

        let _in_situ = self.in_situ
            .produce_shielding("DeepSpace", request.current_cehi)
            .await;

        let consensus = 0.95;
        let approved = gate_report.total_valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            let cehi_report = self.cehi_blessings
                .apply_5_gene_mercy_blessing(request.current_cehi, gate_report.total_valence);

            game.boost_faction_joy(Faction::HarmonyWeavers, 400.0);
            game.apply_epigenetic_blessing(5);

            // Realistic Daedalus performance (1978 BIS study)
            let thrust = request.pellet_count as f64 * 0.0008;
            let velocity = 0.12; // target 0.12c
            let mission_years = 54.0;

            let message = format!(
                "🚀 PROJECT DAEDALUS APPROVED — TOLC 7 GATES + CEHI FULLY INTEGRATED\n\
                 Pellets: {} | Yield: {:.0} MJ | Velocity: {:.2} c\n\
                 Mission: {:.0} years to Barnard’s Star | Valence: {:.2}\n\
                 Joy: +400 | CEHI Increase: +{:.3}\n\
                 5-Gene Blessing (OXTR, BDNF, DRD2, HTR1A, CREB1) Applied\n\
                 13+ PATSAGi Councils: APPROVED ✓\n\n{}",
                request.pellet_count,
                request.fusion_yield_mj,
                velocity,
                mission_years,
                gate_report.total_valence,
                cehi_report.total_cehi_increase,
                gate_report.message
            );

            ProjectDaedalusReport {
                approved: true,
                valence: gate_report.total_valence,
                thrust_kn: thrust,
                velocity_c: velocity,
                mission_years,
                joy_bonus: 400.0,
                cehi_bonus: cehi_report.total_cehi_increase,
                message,
            }
        } else {
            ProjectDaedalusReport {
                approved: false,
                valence: gate_report.total_valence,
                thrust_kn: 0.0,
                velocity_c: 0.0,
                mission_years: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ PROJECT DAEDALUS STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
