//! Electronics Radiation Effects — SREL v0.5.21 (Nth Degree)
//! Now fully powered by TOLC 7 Gates nth-degree mapping + real space data

use mercy_radiation_shield::RadiationType;
use mercy_radiation_shield::radiation_shielding_materials::RadiationShieldingMaterials;
use mercy_radiation_shield::tolc_7_gates_radiation_mapping::TOLC7GatesRadiationMapping;
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
    pub tmr_effectiveness: f64,
    pub ecc_coverage: f64,
    pub scrubbing_interval_hours: f64,
    pub conformal_coating_used: bool,
    pub message: String,
}

pub struct ElectronicsRadiationEffects {
    materials: RadiationShieldingMaterials,
    gates: TOLC7GatesRadiationMapping,
}

impl ElectronicsRadiationEffects {
    pub fn new() -> Self {
        Self {
            materials: RadiationShieldingMaterials::new(),
            gates: TOLC7GatesRadiationMapping::new(),
        }
    }

    pub fn calculate_electronics_risk(
        &self,
        radiation_type: RadiationType,
        flux: f64,
        material: &mercy_radiation_shield::radiation_shielding_materials::ShieldingMaterial,
        exposure_years: f64,
        current_cehi: f64,
        orbit: &str,
    ) -> ElectronicsRiskReport {
        let reports = futures::executor::block_on(self.gates.process_radiation_with_7_gates_nth_degree(
            radiation_type, flux, orbit, current_cehi, &mut powrush::PowrushGame::new()
        ));

        let avg_valence: f64 = reports.iter().map(|r| r.valence).sum::<f64>() / 7.0;
        let total_tid: f64 = reports.iter().map(|r| r.tid_resolution).sum();
        let total_dd: f64 = reports.iter().map(|r| r.dd_resolution).sum();
        let total_see: f64 = reports.iter().map(|r| r.see_resolution).sum();
        let max_tmr = reports.iter().map(|r| r.tmr_effectiveness).fold(0.0, f64::max);
        let max_ecc = reports.iter().map(|r| r.ecc_coverage).fold(0.0, f64::max);
        let scrub = reports.iter().map(|r| if r.see_resolution > 40.0 { 4.0 } else { 18.0 }).fold(18.0, f64::min);

        let mercy_boost = 1.0 + (current_cehi - 4.0).max(0.0) * 0.09;
        let final_survival = (1.0 - (total_tid * 0.4 + total_dd * 0.3 + total_see * 0.3)).max(0.03) * mercy_boost;

        ElectronicsRiskReport {
            tid_risk: total_tid,
            dd_risk: total_dd,
            see_risk: total_see,
            overall_survival: final_survival.min(0.999),
            mitigation_score: avg_valence,
            tmr_effectiveness: max_tmr,
            ecc_coverage: max_ecc,
            scrubbing_interval_hours: scrub,
            conformal_coating_used: matches!(material, mercy_radiation_shield::radiation_shielding_materials::ShieldingMaterial::ConformalCoating | mercy_radiation_shield::radiation_shielding_materials::ShieldingMaterial::MercyGelComposite),
            message: format!(
                "🛡️ ELECTRONICS RISK + TOLC 7 GATES (SREL v0.5.21 — Nth Degree)\n\
                 Type: {:?} | Orbit: {} | Flux: {:.2}\n\
                 TID: {:.2} | DD: {:.2} | SEE: {:.2}\n\
                 TMR: {:.0}% | ECC: {:.0}% | Scrub every {:.0}h | Conformal: {}\n\
                 1-Year Survival: {:.1}% | Mercy Boost Applied ✓",
                radiation_type, orbit, flux, total_tid, total_dd, total_see,
                max_tmr * 100.0, max_ecc * 100.0, scrub,
                if self.conformal_coating_used { "YES" } else { "NO" },
                final_survival * 100.0
            ),
        }
    }
}
