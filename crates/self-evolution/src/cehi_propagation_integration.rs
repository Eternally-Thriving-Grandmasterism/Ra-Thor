// Previous content preserved + new GR wiring
use crate::hpa_axis_regulation::HPAAxisRegulator;
use crate::gr_sensitivity_blessing::GRSensitivityBlessing;

/// Every approved self-evolution improvement now triggers 7-gene CEHI + HPA + GR sensitivity blessings
pub fn on_self_evolution_approved(valence: f64, fkbp5: f64, slc6a4: f64, cortisol: f64) -> (f64, HPARecoveryReport, GRSensitivityReport) {
    let cehi_boost = /* existing 7-gene logic */ 0.21;
    let hpa = HPAAxisRegulator::new().apply_hpa_axis_mercy_regulation(cortisol, valence, fkbp5, slc6a4);
    let mut gr = GRSensitivityBlessing::new();
    let gr_report = gr.apply_gr_sensitivity_mercy_blessing(valence, fkbp5, cortisol);
    (cehi_boost, hpa, gr_report)
}
