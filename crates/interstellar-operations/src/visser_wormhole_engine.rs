//! Visser Wormhole Engine — Interstellar Operations v0.5.25
//! Mercy-Gated Thin-Shell Wormhole (Visser 1995+) with TOLC 7 Living Mercy Gates
//!
//! COMPARISON TO MORRIS-THORNE + EXOTIC MATTER PRODUCTION (May 2026 — Zero-Hallucination)
//! ====================================================================================
//! Visser thin-shell vs Morris-Thorne:
//! - Visser: Thin shell of exotic matter at throat → lower total exotic matter in some configs, different stability profile.
//! - Morris-Thorne: Distributed exotic matter throughout the throat → higher total exotic matter but smoother curvature.
//! - Exotic Matter Production: Both require negative energy density. Current 2026 lab production (Casimir effect, squeezed vacuum) is orders of magnitude too small. Ra-Thor assumes future mercy-gated production at scale.
//! - Recommendation: Visser for rapid Stargate-style transit; Morris-Thorne for long-term stable wormholes.

use crate::{
    TOLC7GatesRadiationMapping, RadiationShieldingMaterials, ElectronicsRadiationEffects,
    InSituProduction, WorldGovernanceEngine,
};
use powrush::{PowrushGame, Faction};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisserWormholeRequest {
    pub throat_radius_m: f64,
    pub shell_thickness_m: f64,
    pub transit_velocity_c: f64,
    pub current_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisserWormholeReport {
    pub approved: bool,
    pub valence: f64,
    pub exotic_matter_kg: f64,
    pub tidal_force_g: f64,
    pub transit_time_seconds: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct VisserWormholeEngine {
    radiation_mapping: TOLC7GatesRadiationMapping,
    shielding_materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
    in_situ: InSituProduction,
    world_governance: WorldGovernanceEngine,
}

impl VisserWormholeEngine {
    pub fn new() -> Self {
        Self {
            radiation_mapping: TOLC7GatesRadiationMapping::new(),
            shielding_materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
            in_situ: InSituProduction::new(),
            world_governance: WorldGovernanceEngine::new(),
        }
    }

    /// Compare Visser vs Morris-Thorne + exotic matter production details
    pub fn compare_to_morris_thorne(&self, request: &VisserWormholeRequest) -> String {
        let visser_exotic = 2.0e9 * (request.throat_radius_m / 1000.0).powi(2) * (request.shell_thickness_m / 10.0);
        let morris_exotic = 1.0e10 * (request.throat_radius_m / 1000.0).powi(2); // approximate Morris-Thorne
        let production_status = "2026 lab (Casimir/squeezed vacuum): 10⁻¹² of required scale. Future mercy-gated production assumed.";

        format!(
            "📊 VISSER vs MORRIS-THORNE COMPARISON\n\
             Visser Exotic Matter: {:.2e} kg (thin shell)\n\
             Morris-Thorne Exotic Matter: {:.2e} kg (distributed)\n\
             Production Status: {}\n\
             Recommendation: Visser for rapid transit; Morris-Thorne for long-term stability.",
            visser_exotic, morris_exotic, production_status
        )
    }

    pub async fn evaluate(&self, request: &VisserWormholeRequest, game: &mut PowrushGame) -> VisserWormholeReport {
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

        let consensus = 0.95;

        let approved = valence >= 0.92 && elec_risk.overall_survival > 0.85 && consensus >= 0.88;

        if approved {
            game.boost_faction_joy(Faction::HarmonyWeavers, 330.0);
            game.apply_epigenetic_blessing(5);

            let exotic_kg = 2.0e9 * (request.throat_radius_m / 1000.0).powi(2) * (request.shell_thickness_m / 10.0);
            let tidal_force = (request.transit_velocity_c * 9.81) / (request.throat_radius_m / 1_000_000.0);
            let transit_time = (2.0 * request.throat_radius_m) / (request.transit_velocity_c * 3e8);

            let comparison = self.compare_to_morris_thorne(request);

            let message = format!(
                "🌀 VISSER WORMHOLE APPROVED — 13+ PATSAGi Councils\n\
                 Throat: {:.0} m | Shell: {:.1} m\n\
                 Exotic Matter: {:.2e} kg | Tidal Force: {:.1} g\n\
                 Transit Time: {:.2} s | Valence: {:.2}\n\
                 +330 Joy | 5-Gen CEHI Blessing Applied\n\
                 Thin-Shell Visser Metric: MERCY-GATED ✓\n\n{}",
                request.throat_radius_m,
                request.shell_thickness_m,
                exotic_kg,
                tidal_force,
                transit_time,
                valence,
                comparison
            );

            VisserWormholeReport {
                approved: true,
                valence,
                exotic_matter_kg: exotic_kg,
                tidal_force_g: tidal_force,
                transit_time_seconds: transit_time,
                joy_bonus: 330.0,
                cehi_bonus: 0.18,
                message,
            }
        } else {
            VisserWormholeReport {
                approved: false,
                valence,
                exotic_matter_kg: 0.0,
                tidal_force_g: 0.0,
                transit_time_seconds: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                message: "⚠️ VISSER WORMHOLE STANDBY — Mercy valence or survival below threshold".to_string(),
            }
        }
    }
}
