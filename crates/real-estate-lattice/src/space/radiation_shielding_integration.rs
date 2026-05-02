//! Radiation Shielding Integration — SREL v0.5.21
//! Mercy-Alchemical • Quantum Swarm • TOLC 7 Gates
//! Merged: Old direct calls + full TOLC7GatesRadiationMapping delegation

use mercy_radiation_shield::RadiationType;
use mercy_radiation_shield::tolc_7_gates_radiation_mapping::TOLC7GatesRadiationMapping;
use powrush::PowrushGame;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadiationIntegrationReport {
    pub transmuted: bool,
    pub energy_recovered: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub valence: f64,
    pub consensus: f64,
    pub message: String,
}

pub struct RadiationShieldingIntegration {
    mapping: TOLC7GatesRadiationMapping,
}

impl RadiationShieldingIntegration {
    pub fn new() -> Self {
        Self {
            mapping: TOLC7GatesRadiationMapping::new(),
        }
    }

    pub async fn process_radiation_event(
        &self,
        radiation_type: RadiationType,
        flux: f64,
        location: &str,
        current_cehi: f64,
        game: &mut PowrushGame,
    ) -> RadiationIntegrationReport {
        let reports = self.mapping
            .process_radiation_with_7_gates(radiation_type, flux, location, current_cehi, game)
            .await;

        let total_energy: f64 = reports.iter().map(|r| r.energy_recovered).sum();
        let total_joy: f64 = reports.iter().map(|r| r.joy_bonus).sum();
        let avg_valence: f64 = reports.iter().map(|r| r.valence).sum::<f64>() / 7.0;

        if avg_valence >= 0.92 {
            let report = RadiationIntegrationReport {
                transmuted: true,
                energy_recovered: total_energy,
                joy_bonus: total_joy,
                cehi_bonus: 0.18,
                valence: avg_valence,
                consensus: 0.88,
                message: format!(
                    "🛡️⚡ RADIATION ALCHEMIZED VIA TOLC 7 GATES (SREL v0.5.21)\n\
                     Type: {:?}\n\
                     Flux: {:.2} → {:.2} usable energy\n\
                     Average Gate Valence: {:.2} | Joy: +{:.1} | CEHI +0.18 (5-gen legacy)\n\
                     13+ PATSAGi Councils: APPROVED ✓\n\
                     All crew + habitat thriving. Radiation transmuted into abundance.",
                    radiation_type, flux, total_energy, avg_valence, total_joy
                ),
            };

            info!("Rathor.ai: Radiation successfully mercy-alchemized via all 7 TOLC Gates at {}", location);
            report
        } else {
            RadiationIntegrationReport {
                transmuted: false,
                energy_recovered: 0.0,
                joy_bonus: 0.0,
                cehi_bonus: 0.0,
                valence: avg_valence,
                consensus: 0.80,
                message: format!(
                    "🛡️ MERCY-GATED — Radiation shielding activated (average valence {:.2})\n\
                     Transmutation threshold not met. Safe mode engaged. Additional mercy alignment recommended.",
                    avg_valence
                ),
            }
        }
    }
}
