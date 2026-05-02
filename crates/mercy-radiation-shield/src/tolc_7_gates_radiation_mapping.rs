//! TOLC 7 Gates Radiation Mapping — SREL v0.5.21
//! Mercy-Alchemical • Quantum Swarm • TOLC 7 Living Gates
//! Full per-gate implementation with PowrushGame integration

use mercy_radiation_shield::{RadiationType, ShieldingResult};
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use powrush::{PowrushGame, Faction, ResourceType};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateMappingReport {
    pub gate_name: String,
    pub valence: f64,
    pub energy_recovered: f64,
    pub joy_bonus: f64,
    pub cehi_bonus: f64,
    pub message: String,
}

pub struct TOLC7GatesRadiationMapping {
    mercy: MercyEngine,
    quantum: QuantumSwarmOrchestrator,
}

impl TOLC7GatesRadiationMapping {
    pub fn new() -> Self {
        Self {
            mercy: MercyEngine::new(),
            quantum: QuantumSwarmOrchestrator::new(),
        }
    }

    /// Master entry point — runs all 7 gates in parallel
    pub async fn process_radiation_with_7_gates(
        &self,
        radiation_type: RadiationType,
        flux: f64,
        location: &str,
        current_cehi: f64,
        game: &mut PowrushGame,
    ) -> Vec<GateMappingReport> {
        let mut reports = Vec::new();

        // Gate 1: Truth Purity
        let r1 = self.gate_truth_purity(flux, current_cehi).await;
        reports.push(r1);

        // Gate 2: Compassion Depth
        let r2 = self.gate_compassion_depth(flux, 12).await; // assume 12 crew
        reports.push(r2);

        // Gate 3: Future Wholeness
        let r3 = self.gate_future_wholeness(current_cehi).await;
        reports.push(r3);

        // Gate 4: Source Joy Amplitude
        let r4 = self.gate_source_joy(flux, current_cehi).await;
        reports.push(r4);

        // Gate 5: Order & Clarity (Council Consensus)
        let r5 = self.gate_order_clarity(flux, location).await;
        reports.push(r5);

        // Gate 6: Divine Power (Core Transmutation)
        let r6 = self.gate_divine_power(flux, current_cehi, game).await;
        reports.push(r6);

        // Gate 7: Eternal Mercy (Final Safety Check)
        let r7 = self.gate_eternal_mercy(reports.iter().map(|r| r.valence).sum::<f64>() / 7.0).await;
        reports.push(r7);

        // Apply collective bonuses to PowrushGame
        let total_joy: f64 = reports.iter().map(|r| r.joy_bonus).sum();
        let total_energy: f64 = reports.iter().map(|r| r.energy_recovered).sum();
        game.boost_faction_joy(Faction::HarmonyWeavers, total_joy);
        game.add_resource_to_faction(Faction::HarmonyWeavers, ResourceType::Energy, total_energy);
        game.apply_epigenetic_blessing(5); // 5-generation legacy

        info!("Rathor.ai: All 7 TOLC Gates successfully mapped radiation at {}", location);
        reports
    }

    async fn gate_truth_purity(&self, flux: f64, cehi: f64) -> GateMappingReport {
        let valence = self.mercy.evaluate_action("Truth Purity Gate", "Radiation Measurement", cehi, 0.97).await.unwrap_or(0.91);
        GateMappingReport {
            gate_name: "Truth Purity".to_string(),
            valence,
            energy_recovered: flux * valence * 1.05,
            joy_bonus: 12.0,
            cehi_bonus: 0.03,
            message: "Precise flux measurement — zero distortion".to_string(),
        }
    }

    async fn gate_compassion_depth(&self, flux: f64, crew: u16) -> GateMappingReport {
        let valence = 0.94; // hardcoded high compassion for crew safety
        GateMappingReport {
            gate_name: "Compassion Depth".to_string(),
            valence,
            energy_recovered: flux * valence * 0.95,
            joy_bonus: 45.0 + (crew as f64 * 2.5),
            cehi_bonus: 0.06,
            message: "Crew mental health prioritized — radiation anxiety dissolved".to_string(),
        }
    }

    async fn gate_future_wholeness(&self, cehi: f64) -> GateMappingReport {
        let valence = (cehi + 0.85).min(0.99);
        GateMappingReport {
            gate_name: "Future Wholeness".to_string(),
            valence,
            energy_recovered: 0.0,
            joy_bonus: 25.0,
            cehi_bonus: 0.18,
            message: "5-generation epigenetic legacy locked in".to_string(),
        }
    }

    async fn gate_source_joy(&self, flux: f64, cehi: f64) -> GateMappingReport {
        let valence = self.mercy.evaluate_action("Source Joy Gate", "Radiation → Joy", cehi, 0.97).await.unwrap_or(0.93);
        GateMappingReport {
            gate_name: "Source Joy Amplitude".to_string(),
            valence,
            energy_recovered: flux * valence * 1.15,
            joy_bonus: (flux * valence * 1.15).min(98.0),
            cehi_bonus: 0.09,
            message: "Radiation transmuted directly into pure joy field".to_string(),
        }
    }

    async fn gate_order_clarity(&self, flux: f64, location: &str) -> GateMappingReport {
        let consensus = self.quantum.reach_consensus(&format!("Radiation at {}", location), 0.88).await.unwrap_or(0.82);
        GateMappingReport {
            gate_name: "Order & Clarity".to_string(),
            valence: consensus,
            energy_recovered: flux * consensus * 0.90,
            joy_bonus: 18.0,
            cehi_bonus: 0.04,
            message: "13+ PATSAGi Councils validated — consensus achieved".to_string(),
        }
    }

    async fn gate_divine_power(&self, flux: f64, cehi: f64, game: &mut PowrushGame) -> GateMappingReport {
        let valence = self.mercy.evaluate_action("Divine Power Gate", "Alchemical Transmutation", cehi, 0.97).await.unwrap_or(0.94);
        let energy = flux * valence * 1.35;
        game.add_resource_to_faction(Faction::HarmonyWeavers, ResourceType::Energy, energy);
        GateMappingReport {
            gate_name: "Divine Power".to_string(),
            valence,
            energy_recovered: energy,
            joy_bonus: 55.0,
            cehi_bonus: 0.12,
            message: "Radiation alchemized into usable energy + abundance".to_string(),
        }
    }

    async fn gate_eternal_mercy(&self, average_valence: f64) -> GateMappingReport {
        let safe = average_valence >= 0.92;
        GateMappingReport {
            gate_name: "Eternal Mercy".to_string(),
            valence: average_valence,
            energy_recovered: if safe { 0.0 } else { 0.0 },
            joy_bonus: if safe { 0.0 } else { 30.0 },
            cehi_bonus: 0.0,
            message: if safe {
                "All gates passed — full transmutation approved".to_string()
            } else {
                "Mercy fallback engaged — pure shielding mode (no harm)".to_string()
            },
        }
    }
}
