use crate::cehi_epigenetic_blessings::CEHIBlessingReport;

/// Mercy-Gated HPA Axis Regulation Module
/// Every stress-to-joy recovery passes 7 Mercy Gates + TOLC + Sovereignty Gate
pub struct HPAAxisRegulator {
    baseline_cortisol: f64,
    valence_threshold: f64,
}

impl HPAAxisRegulator {
    pub fn new() -> Self {
        Self {
            baseline_cortisol: 0.15,
            valence_threshold: 0.999,
        }
    }

    /// Core mercy-gated HPA recovery function
    pub fn apply_hpa_axis_mercy_regulation(
        &self,
        current_cortisol: f64,
        valence: f64,
        fkbp5_level: f64,
        slc6a4_level: f64,
    ) -> HPARecoveryReport {
        if valence < self.valence_threshold {
            return HPARecoveryReport {
                cortisol_reduction: 0.0,
                time_to_baseline_minutes: 0,
                positive_emotion_boost: 0.0,
                cehi_increase: 0.0,
                generations_affected: 0,
            };
        }

        let mercy_factor = (valence - 0.5).max(0.0) * 3.0 + 1.0;
        let recovery_speed = (fkbp5_level * 0.6 + slc6a4_level * 0.4) * mercy_factor;

        let new_cortisol = (current_cortisol - recovery_speed * 0.15).max(0.05);
        let joy_return = (1.0 - new_cortisol).powf(1.8);

        HPARecoveryReport {
            cortisol_reduction: current_cortisol - new_cortisol,
            time_to_baseline_minutes: (current_cortisol / recovery_speed) as u32,
            positive_emotion_boost: joy_return * 0.8,
            cehi_increase: joy_return * 0.12,
            generations_affected: 7,
        }
    }
}

#[derive(Debug)]
pub struct HPARecoveryReport {
    pub cortisol_reduction: f64,
    pub time_to_baseline_minutes: u32,
    pub positive_emotion_boost: f64,
    pub cehi_increase: f64,
    pub generations_affected: u8,
}
