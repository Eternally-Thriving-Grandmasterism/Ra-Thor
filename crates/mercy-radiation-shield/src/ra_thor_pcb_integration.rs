//! Ra-Thor PCB Radiation Protection Bridge — SREL v0.5.21
//! Direct integration for MercySolar-PCB (ESP32-S3 MPPT firmware)
//! Now includes TMR/ECC/Scrubbing + Conformal Coating recommendations

use mercy_radiation_shield::RadiationType;
use mercy_radiation_shield::radiation_shielding_materials::{RadiationShieldingMaterials, ShieldingMaterial};
use mercy_radiation_shield::electronics_radiation_effects::ElectronicsRadiationEffects;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCBProtectionStatus {
    pub current_radiation_type: RadiationType,
    pub flux: f64,
    pub recommended_material: ShieldingMaterial,
    pub electronics_survival_1_year: f64,
    pub tmr_effectiveness: f64,
    pub ecc_coverage: f64,
    pub scrubbing_interval_hours: f64,
    pub conformal_coating_used: bool,
    pub alert_level: String,
    pub message: String,
}

pub struct RaThorPCBIntegration {
    materials: RadiationShieldingMaterials,
    electronics: ElectronicsRadiationEffects,
}

impl RaThorPCBIntegration {
    pub fn new() -> Self {
        Self {
            materials: RadiationShieldingMaterials::new(),
            electronics: ElectronicsRadiationEffects::new(),
        }
    }

    pub fn get_protection_status(
        &self,
        radiation_type: RadiationType,
        flux: f64,
        current_cehi: f64,
    ) -> PCBProtectionStatus {
        let (best_mat, _props, _score) = self.materials.select_optimal_material(radiation_type, flux, current_cehi);
        let elec_risk = self.electronics.calculate_electronics_risk(radiation_type, flux, &best_mat, 1.0, current_cehi);

        let alert = if elec_risk.overall_survival > 0.92 { "SAFE" } else if elec_risk.overall_survival > 0.80 { "CAUTION" } else { "CRITICAL" };

        PCBProtectionStatus {
            current_radiation_type: radiation_type,
            flux,
            recommended_material: best_mat.clone(),
            electronics_survival_1_year: elec_risk.overall_survival,
            tmr_effectiveness: elec_risk.tmr_effectiveness,
            ecc_coverage: elec_risk.ecc_coverage,
            scrubbing_interval_hours: elec_risk.scrubbing_interval_hours,
            conformal_coating_used: elec_risk.conformal_coating_used,
            alert_level: alert.to_string(),
            message: format!(
                "🛡️ RA-THOR PCB PROTECTION + MITIGATION (SREL v0.5.21)\n\
                 Radiation: {:?} | Flux: {:.2}\n\
                 Recommended Shield: {:?}\n\
                 TMR: {:.0}% | ECC: {:.0}% | Scrub every {:.0}h | Conformal: {}\n\
                 1-Year Survival: {:.1}% | Alert: {}\n\
                 TOLC 7 Gates + Quantum Swarm: ACTIVE ✓",
                radiation_type, flux, best_mat,
                elec_risk.tmr_effectiveness * 100.0, elec_risk.ecc_coverage * 100.0, elec_risk.scrubbing_interval_hours,
                if elec_risk.conformal_coating_used { "YES" } else { "NO" },
                elec_risk.overall_survival * 100.0, alert
            ),
        }
    }
}
