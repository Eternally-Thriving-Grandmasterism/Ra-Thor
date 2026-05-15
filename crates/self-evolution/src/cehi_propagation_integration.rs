// Previous content preserved + new HPA wiring
use crate::hpa_axis_regulation::HPAAxisRegulator;

// ... existing CEHI integration code ...

/// Every approved self-evolution improvement now triggers both 7-gene CEHI + HPA recovery
pub fn on_self_evolution_approved(valence: f64, fkbp5: f64, slc6a4: f64) -> (f64, HPARecoveryReport) {
    let cehi_boost = /* existing 7-gene logic */;
    let hpa = HPAAxisRegulator::new().apply_hpa_axis_mercy_regulation(0.8, valence, fkbp5, slc6a4);
    (cehi_boost, hpa)
}
