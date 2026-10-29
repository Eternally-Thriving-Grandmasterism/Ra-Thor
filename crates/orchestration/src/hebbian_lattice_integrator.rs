// crates/orchestration/src/hebbian_lattice_integrator.rs
// Ra-Thor™ Hebbian Lattice Integrator — Full Plasticity Core Wired into Unified Sovereign Energy Lattice + Self-Improvement
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Real-time, mercy-gated, novelty-driven Hebbian learning on lattice state vectors
// Fully integrated with STDPHebbianPlasticityCore, UnifiedSovereignEnergyLatticeCore, SelfImprovementCore
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use crate::stdp_hebbian_plasticity_core::STDPHebbianPlasticityCore;
use crate::unified_sovereign_energy_lattice_core::UnifiedSovereignEnergyLatticeCore;
use crate::self_improvement_core::SelfImprovementCore;
use ra_thor_mercy::MercyError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct HebbianLatticeReport {
    pub novelty_boost: f64,
    pub updated_valence: f64,
    pub active_components: usize,
    pub bloom_intensity: f64,
}

pub struct HebbianLatticeIntegrator {
    plasticity: STDPHebbianPlasticityCore,
    lattice: UnifiedSovereignEnergyLatticeCore,
    self_improvement: SelfImprovementCore,
}

impl HebbianLatticeIntegrator {
    pub fn new() -> Self {
        Self {
            plasticity: STDPHebbianPlasticityCore::new(),
            lattice: UnifiedSovereignEnergyLatticeCore::new(),
            self_improvement: SelfImprovementCore::new(),
        }
    }

    /// Main integration method — runs one full Hebbian step on current lattice state
    pub async fn run_hebbian_lattice_step(
        &mut self,
        context: &str,
        current_valence: f64,
    ) -> Result<HebbianLatticeReport, MercyError> {
        // 1. Get current lattice state
        let lattice_report = self.lattice.orchestrate_energy_lattice(context, None).await?;

        // 2. Run full plasticity on key lattice signals
        let mut total_novelty = 0.0;

        // Process mercy valence signal
        let (novelty1, _) = self.plasticity.process_timestep(
            "lattice_valence",
            current_valence,
            current_valence,
            10.0,
        );
        total_novelty += novelty1;

        // Process bloom intensity
        let (novelty2, _) = self.plasticity.process_timestep(
            "bloom_intensity",
            lattice_report.bloom_intensity,
            current_valence,
            10.0,
        );
        total_novelty += novelty2;

        // Apply Sanger's multi-component extraction on technology scores (example)
        let tech_scores: Vec<f64> = vec![
            lattice_report.overall_system_health,
            lattice_report.projected_25yr_thriving,
            current_valence,
        ];
        self.plasticity.apply_sangers_rule("lattice_tech", &tech_scores, 0, current_valence);
        self.plasticity.apply_sangers_rule("lattice_tech", &tech_scores, 1, current_valence);

        // 3. Feed novelty back into self-improvement
        if total_novelty > 0.05 {
            let _ = self.self_improvement.run_self_improvement_cycle().await;
        }

        Ok(HebbianLatticeReport {
            novelty_boost: total_novelty,
            updated_valence: current_valence * (1.0 + total_novelty * 0.3),
            active_components: 2,
            bloom_intensity: lattice_report.bloom_intensity,
        })
    }
}
