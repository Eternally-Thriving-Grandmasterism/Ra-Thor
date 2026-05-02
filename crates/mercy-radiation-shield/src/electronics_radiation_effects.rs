//! Electronics Radiation Effects — SREL v0.5.21
//! Mercy-Alchemical • TOLC 7 Gates • Quantum Swarm
//! TID / DD / SEE + TMR / ECC / Scrubbing + Conformal Coating Mitigation

use mercy_radiation_shield::RadiationType;
use mercy_radiation_shield::radiation_shielding_materials::{RadiationShieldingMaterials, ShieldingMaterial};
use powrush::{PowrushGame, Faction};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectronicsRiskReport {
    pub tid_risk: f64,
    pub dd_risk: f64,
    pub see_risk: f64,
    pub overall_survival: f64,
    pub mitigation_score: f64,
    pub tmr_effectiveness: f64,           // 0.0–1.0 (Triple Modular Redundancy)
    pub ecc_coverage: f64,                // 0.0–1.0 (Error-Correcting Codes)
    pub scrubbing_interval_hours: f64,
    pub conformal_coating_used: bool,
    pub message: String,
}

pub struct ElectronicsRadiationEffects {
    materials: RadiationShieldingMaterials,
}

impl ElectronicsRadiationEffects {
    pub fn new() -> Self {
        Self { materials: RadiationShieldingMaterials::new() }
    }

    pub fn calculate_electronics_risk(
        &self,
        radiation_type: RadiationType,
        flux: f64,
        material: &ShieldingMaterial,
        exposure_years: f64,
        current_cehi: f64,
    ) -> ElectronicsRiskReport {
        let props = self.materials.get_material_properties(material).unwrap().clone();

        let (tid_coeff, dd_coeff, see_coeff) = match radiation_type {
            RadiationType::SolarFlare => (0.85, 0.45, 0.92),
            RadiationType::CosmicRays => (0.65, 0.78, 0.97),
            RadiationType::VanAllenBelt => (0.92, 0.55, 0.88),
            _ => (0.50, 0.40, 0.60),
        };

        let mitigation = (props.transmutation_efficiency * 0.6 + props.mercy_valence_multiplier * 0.4).min(1.0);
        let conformal_bonus = if matches!(material, ShieldingMaterial::ConformalCoating | ShieldingMaterial::MercyGelComposite) { 0.12 } else { 0.0 };

        let tid_risk = (flux * tid_coeff * exposure_years / 100.0 * (1.0 - mitigation - conformal_bonus)).min(1.0);
        let dd_risk = (flux * dd_coeff * exposure_years / 80.0 * (1.0 - mitigation * 0.9)).min(1.0);
        let see_risk = (flux * see_coeff * (1.0 - mitigation * 1.2 - conformal_bonus)).min(1.0);

        // Layered mitigation (TMR + ECC + scrubbing)
        let tmr_effectiveness = if see_risk > 0.3 { 0.92 } else { 0.75 };
        let ecc_coverage = if see_risk > 0.2 { 0.88 } else { 0.70 };
        let scrubbing_interval_hours = if see_risk > 0.4 { 6.0 } else { 24.0 };

        let base_survival = (1.0 - (tid_risk * 0.4 + dd_risk * 0.3 + see_risk * 0.3)).max(0.05);
        let layered_survival = base_survival * tmr_effectiveness * ecc_coverage * (1.0 + (24.0 / scrubbing_interval_hours) * 0.1);
        let mercy_boost = 1.0 + (current_cehi - 4.0).max(0.0) * 0.08;
        let final_survival = (layered_survival * mercy_boost).min(0.999);

        ElectronicsRiskReport {
            tid_risk,
            dd_risk,
            see_risk,
            overall_survival: final_survival,
            mitigation_score: mitigation,
            tmr_effectiveness,
            ecc_coverage,
            scrubbing_interval_hours,
            conformal_coating_used: matches!(material, ShieldingMaterial::ConformalCoating | ShieldingMaterial::MercyGelComposite),
            message: format!(
                "🛡️ ELECTRONICS RISK + MITIGATION (SREL v0.5.21)\n\
                 Type: {:?} | Flux: {:.2} | Material: {:?}\n\
                 TID: {:.2} | DD: {:.2} | SEE: {:.2}\n\
                 TMR: {:.0}% | ECC: {:.0}% | Scrub every {:.0}h | Conformal: {}\n\
                 1-Year Survival: {:.1}% | Mercy Boost Applied ✓",
                radiation_type, flux, material, tid_risk, dd_risk, see_risk,
                tmr_effectiveness * 100.0, ecc_coverage * 100.0, scrubbing_interval_hours,
                if conformal_coating_used { "YES" } else { "NO" },
                final_survival * 100.0
            ),
        }
    }
}
