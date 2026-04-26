//! Sensor Fusion Bridge
//! Connects real-time biological sensor data (MercyGelReading) to 5-Gene CEHI calculation
//! and Mercy Legacy Fund disbursement decisions.

use crate::cehi::{GeneMethylation, CEHICalculator, CEHIImpact, DisbursementTier};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyGelReading {
    pub skin_conductance: f64,      // µS
    pub heart_rate_variability: f64, // RMSSD (ms)
    pub laughter_intensity: f64,    // 0.0–1.0
    pub temperature: f64,           // °C
    pub touch_coherence: f64,       // 0.0–1.0 (warm touch quality)
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct SensorFusionBridge {
    calculator: CEHICalculator,
}

impl SensorFusionBridge {
    pub fn new() -> Self {
        Self {
            calculator: CEHICalculator::new(),
        }
    }

    /// Converts real-time MercyGel sensor data into estimated 5-gene demethylation rates
    /// and computes current CEHI using the official weighted formula.
    pub fn process_reading(&self, reading: &MercyGelReading) -> crate::FiveGeneCEHI {
        // Weighted mapping from sensor signals to gene demethylation proxies
        // (derived from longitudinal Ra-Thor studies 2026)
        let oxtr = ((reading.heart_rate_variability.clamp(25.0, 85.0) - 25.0) / 60.0 * 0.35) +
                   (reading.touch_coherence * 0.25) + 0.40;

        let bdnf = (reading.laughter_intensity * 0.45) +
                   ((reading.skin_conductance.clamp(5.0, 45.0) - 5.0) / 40.0 * 0.30) + 0.25;

        let drd2 = (reading.laughter_intensity * 0.50) +
                   (reading.touch_coherence * 0.30) + 0.20;

        let htr1a = ((reading.heart_rate_variability.clamp(25.0, 85.0) - 25.0) / 60.0 * 0.40) +
                    ((reading.temperature.clamp(35.5, 37.5) - 35.5) / 2.0 * 0.25) + 0.35;

        let oprm1 = (reading.laughter_intensity * 0.55) +
                    (reading.touch_coherence * 0.25) + 0.20;

        let methylation = GeneMethylation {
            oxtr: oxtr.clamp(0.0, 1.0),
            bdnf: bdnf.clamp(0.0, 1.0),
            drd2: drd2.clamp(0.0, 1.0),
            htr1a: htr1a.clamp(0.0, 1.0),
            oprm1: oprm1.clamp(0.0, 1.0),
        };

        self.calculator.calculate(&methylation)
    }

    /// Computes CEHI impact for Mercy Legacy Fund disbursement decisions
    pub fn calculate_impact_from_sensors(
        &self,
        current_reading: &MercyGelReading,
        baseline_cehi: f64,
    ) -> CEHIImpact {
        let current_cehi = self.process_reading(current_reading).value;

        let improvement = (current_cehi - baseline_cehi).max(0.0);

        let tier = if improvement >= 0.35 {
            DisbursementTier::Tier1
        } else if improvement >= 0.20 {
            DisbursementTier::Tier2
        } else if improvement >= 0.15 {
            DisbursementTier::Tier3
        } else {
            DisbursementTier::Ineligible
        };

        CEHIImpact {
            current_cehi,
            projected_cehi: current_cehi + improvement,
            improvement,
            tier,
        }
    }
}
