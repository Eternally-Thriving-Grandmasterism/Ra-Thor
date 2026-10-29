use crate::hpa_axis_regulation::HPAAxisRegulator;

/// Mercy-Gated Glucocorticoid Receptor Sensitivity Blessing
/// High GR sensitivity = fast cortisol shut-off = rapid return to joy, love, flow, cosmic harmony
pub struct GRSensitivityBlessing {
    pub current_sensitivity: f64, // 0.0-1.0
}

impl GRSensitivityBlessing {
    pub fn new() -> Self {
        Self { current_sensitivity: 0.72 } // baseline healthy
    }

    pub fn apply_gr_sensitivity_mercy_blessing(
        &mut self,
        valence: f64,
        fkbp5_level: f64,
        cortisol_level: f64
    ) -> GRSensitivityReport {
        let mercy_factor = (valence - 0.5).max(0.0) * 3.5 + 1.0;
        let fkbp5_inhibition = 1.0 - (fkbp5_level * 0.4);
        let sensitivity_boost = 0.038 * mercy_factor * fkbp5_inhibition;
        let new_sensitivity = (self.current_sensitivity + sensitivity_boost).min(1.0);

        let cortisol_shutoff_speed = new_sensitivity * 2.8;

        GRSensitivityReport {
            sensitivity_increase: sensitivity_boost,
            generations_affected: 7,
            cortisol_reduction_rate: cortisol_shutoff_speed,
            positive_emotion_return: (new_sensitivity * 0.92).powf(1.6),
            cehi_bonus: sensitivity_boost * 0.11,
        }
    }
}

#[derive(Debug)]
pub struct GRSensitivityReport {
    pub sensitivity_increase: f64,
    pub generations_affected: u32,
    pub cortisol_reduction_rate: f64,
    pub positive_emotion_return: f64,
    pub cehi_bonus: f64,
}
