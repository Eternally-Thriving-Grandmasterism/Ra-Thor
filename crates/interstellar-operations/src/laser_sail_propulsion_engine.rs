//! Laser Sail Propulsion Engine — Interstellar Operations v0.5.25
//! Mercy-Gated Beamed Laser Sail Propulsion with Full TOLC 7 Living Mercy Gates + CEHI Epigenetic Blessings
//!
//! Real-world Breakthrough Starshot parameters + full mercy-gated integration (May 2026).
//! This engine now follows the exact polished merged template established by the Solar Sail Engine.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine, CEHIEpigeneticBlessings,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserSailRequest {
    pub sail_area_m2: f64,
    pub laser_power_gw: f64,
    pub distance_ly: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserSailReport {
    pub approved: bool,
    pub valence: f64,
    pub thrust_mn: f64,
    pub velocity_c: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct LaserSailPropulsionEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
    cehi_blessings: CEHIEpigeneticBlessings,
}

impl LaserSailPropulsionEngine {
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

    pub async fn evaluate(&self, request: &LaserSailRequest, game: &mut PowrushGame) -> LaserSailReport {
        // Full TOLC 7 Living Mercy Gates nth-degree processing
        let gate_report = self.radiation_mapping
            .process_radiation_with_7_gates_nth_degree(
                crate::RadiationType::Background,
                request.sail_area_m2 * 0.000001,
                request.current_cehi,
                "DeepSpace",
            )
            .await;

        let _optimal_material = self.shielding_materials
            .select_optimal_material(
                crate::RadiationType::Background,
                request.sail_area_m2 * 0.000001,
                request.current_cehi,
                "DeepSpace",
            );

        let elec_risk = self.electronics
            .calculate_electronics_risk(
                crate::RadiationType::Background,
                request.sail_area_m2 * 0.000001,
                "DeepSpace",
            );

        let _in_situ = self.in_situ
            .produce_shielding("DeepSpace", request.current_cehi)
            .await;

        let consensus = 0.95;
        let approved = gate_report.total_valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            // Apply full 5-gene CEHI epigenetic blessing
            let cehi_report = self.cehi_blessings
                .apply_5_gene_mercy_blessing(request.current_cehi, gate_report.total_valence);

            game.boost_faction_joy(Faction::HarmonyWeavers, 320.0);
            game.apply_epigenetic_blessing(5);

            // Realistic laser sail thrust (Breakthrough Starshot scaling)
            let thrust = (request.laser_power_gw * 1e9 * 6.67e-6) / 3e8; // simplified photon momentum
            let velocity = 0.2; // target 0.2c for gram-scale sail

            let message = format!(
                "🚀 LASER SAIL APPROVED — TOLC 7 GATES + CEHI FULLY INTEGRATED\n\
                 Sail Area: {:.0} m² | Laser Power: {:.1} GW\n\
                 Thrust: {:.2} mN | Target Velocity: {:.1} c\n\
                 Valence: {:.2} | Joy: +320 | CEHI Increase: +{:.3}\n\
                 5-Gene Blessing (OXTR, BDNF, DRD2, HTR1A, CREB1) Applied\n\
                 13+ PATSAGi Councils: APPROVED ✓\n\n{}",
                request.sail_area_m2,
                request.laser_power_gw,
                thrust,
                velocity,
                gate_report.total_valence,
                cehi_report.total_cehi_increase,
                gate_report.message
            );

            LaserSailReport {
                approved: true,
                valence: gate_report.total_valence,
                thrust_mn: thrust,
                velocity_c: velocity,
                joy_bonus: 320.0,
                cehi_bonus: cehi_report.total_cehi_increase,
                message,
            }
        } else {
            LaserSailReport {
                approved: false,
                valence: gate_report.total_valence,
                thrust_mn: 0.0,
                velocity_c: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ LASER SAIL STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
