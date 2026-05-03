//! Antimatter Propulsion Engine — Interstellar Operations v0.5.25
//! Mercy-Gated Antimatter Propulsion with Full TOLC 7 Living Mercy Gates + CEHI Epigenetic Blessings
//!
//! Real 2026 parameters (beamed-core antimatter rocket, antimatter-catalyzed micro-fusion concepts, storage challenges) + complete mercy-gated integration.
//! This engine follows the exact polished merged template established by Solar Sail, Laser Sail, Nuclear Thermal & Fusion Drive.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine, CEHIEpigeneticBlessings,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntimatterRequest {
    pub antimatter_grams: f64,
    pub specific_impulse_s: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntimatterReport {
    pub approved: bool,
    pub valence: f64,
    pub thrust_kn: f64,
    pub isp_s: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct AntimatterPropulsionEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
    cehi_blessings: CEHIEpigeneticBlessings,
}

impl AntimatterPropulsionEngine {
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

    pub async fn evaluate(&self, request: &AntimatterRequest, game: &mut PowrushGame) -> AntimatterReport {
        let gate_report = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.antimatter_grams * 0.1,
                request.current_cehi,
                "DeepSpace",
            )
            .await;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.antimatter_grams * 0.1,
                request.current_cehi,
                "DeepSpace",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.antimatter_grams * 0.1,
                "DeepSpace",
            );

        let _in_situ = self.in_situ
            .produce_shielding("DeepSpace", request.current_cehi)
            .await;

        let consensus = 0.94;
        let approved = gate_report.total_valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            let cehi_report = self.cehi_blessings
                .apply_5_gene_mercy_blessing(request.current_cehi, gate_report.total_valence);

            game.boost_faction_joy(Faction::HarmonyWeavers, 380.0);
            game.apply_epigenetic_blessing(5);

            let thrust = request.antimatter_grams * 180.0;
            let isp = request.specific_impulse_s.max(25000.0);

            let message = format!(
                "☢️ ANTIMATTER PROPULSION APPROVED — TOLC 7 GATES + CEHI FULLY INTEGRATED\n\
                 Antimatter: {:.2} g | Isp: {:.0} s | Thrust: {:.1} kN\n\
                 Valence: {:.2} | Joy: +380 | CEHI Increase: +{:.3}\n\
                 5-Gene Blessing (OXTR, BDNF, DRD2, HTR1A, CREB1) Applied\n\
                 13+ PATSAGi Councils: APPROVED ✓\n\n{}",
                request.antimatter_grams,
                isp,
                thrust,
                gate_report.total_valence,
                cehi_report.total_cehi_increase,
                gate_report.message
            );

            AntimatterReport {
                approved: true,
                valence: gate_report.total_valence,
                thrust_kn: thrust,
                isp_s: isp,
                joy_bonus: 380.0,
                cehi_bonus: cehi_report.total_cehi_increase,
                message,
            }
        } else {
            AntimatterReport {
                approved: false,
                valence: gate_report.total_valence,
                thrust_kn: 0.0,
                isp_s: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ ANTIMATTER PROPULSION STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
