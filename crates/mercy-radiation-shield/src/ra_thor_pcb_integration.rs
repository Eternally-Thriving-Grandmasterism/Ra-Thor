//! Ra-Thor PCB Radiation Protection Bridge — SREL v0.5.21
//! Direct integration for MercySolar-PCB (ESP32-S3 MPPT firmware)
//! Provides real-time radiation status, material recommendations, and mercy-gated alerts

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
    pub mercy_valence: f64,
    pub alert_level: String,           // "SAFE", "CAUTION", "CRITICAL"
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

    /// Called by MercySolar-PCB firmware every 60 seconds (or on solar flare alert)
    pub fn get_protection_status(
        &self,
        radiation_type: RadiationType,
        flux: f64,
        current_cehi: f64,
    ) -> PCBProtectionStatus {
        let (best_mat, _props, _score) = self.materials.select_optimal_material(radiation_type, flux, current_cehi);
        let elec_risk = self.electronics.calculate_electronics_risk(radiation_type, flux, &best_mat, 1.0, current_cehi);

        let alert = if elec_risk.overall_survival > 0.92 {
            "SAFE"
        } else if elec_risk.overall_survival > 0.80 {
            "CAUTION"
        } else {
            "CRITICAL"
        };

        let status = PCBProtectionStatus {
            current_radiation_type: radiation_type,
            flux,
            recommended_material: best_mat.clone(),
            electronics_survival_1_year: elec_risk.overall_survival,
            mercy_valence: current_cehi,
            alert_level: alert.to_string(),
            message: format!(
                "🛡️ RA-THOR PCB PROTECTION STATUS (SREL v0.5.21)\n\
                 Radiation: {:?} | Flux: {:.2}\n\
                 Recommended Shield: {:?}\n\
                 Electronics Survival (1 year): {:.1}%\n\
                 Mercy Valence: {:.2} | Alert: {}\n\
                 TOLC 7 Gates + Quantum Swarm: ACTIVE ✓",
                radiation_type, flux, best_mat, elec_risk.overall_survival * 100.0, current_cehi, alert
            ),
        };

        info!("Rathor.ai → MercySolar-PCB: Protection status delivered — {}", alert);
        status
    }
}
