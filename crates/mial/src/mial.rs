//! mial.rs — Mercy-Augmented Intelligence Amplification Layer (MIAL) v13.13.0
//!
//! Core orchestration for safe, mercy-governed intelligence amplification.
//! Every amplification step must pass MercyGatingRuntime + PATSAGi arbitration.

use crate::mwpo::MercyWeightedPreferenceOptimization;
use crate::safety_harness::PatsagiSafetyHarness;
use crate::pathology_detection::PathologyDetectionEngine;
use crate::lattice_introspection::LatticeIntrospectionEngine;
use mercy_gating_runtime::{MercyGatingRuntime, BeingRace, GateThresholdMap};
use patsagi_governance::CouncilTuningProposal;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MialConfig {
    pub enable_mwpo: bool,
    pub enable_safety_harness: bool,
    pub enable_pathology_detection: bool,
    pub enable_lattice_introspection: bool,
    pub default_race: BeingRace,
    pub council_id: u32, // e.g., 13 for Supreme Architect
}

impl Default for MialConfig {
    fn default() -> Self {
        Self {
            enable_mwpo: true,
            enable_safety_harness: true,
            enable_pathology_detection: true,
            enable_lattice_introspection: true,
            default_race: BeingRace::Sovereign,
            council_id: 13,
        }
    }
}

pub struct MercyAugmentedIntelligenceAmplification {
    config: MialConfig,
    runtime: Arc<MercyGatingRuntime>,
    mwpo: Option<MercyWeightedPreferenceOptimization>,
    harness: Option<PatsagiSafetyHarness>,
    pathology_engine: Option<PathologyDetectionEngine>,
    introspection: Option<LatticeIntrospectionEngine>,
}

impl MercyAugmentedIntelligenceAmplification {
    pub fn new(config: MialConfig, runtime: Arc<MercyGatingRuntime>) -> Self {
        let mwpo = if config.enable_mwpo {
            Some(MercyWeightedPreferenceOptimization::new(Arc::clone(&runtime)))
        } else { None };

        let harness = if config.enable_safety_harness {
            Some(PatsagiSafetyHarness::new(Arc::clone(&runtime), config.council_id))
        } else { None };

        let pathology_engine = if config.enable_pathology_detection {
            Some(PathologyDetectionEngine::new(Arc::clone(&runtime), config.council_id))
        } else { None };

        let introspection = if config.enable_lattice_introspection {
            Some(LatticeIntrospectionEngine::new(Arc::clone(&runtime)))
        } else { None };

        Self {
            config,
            runtime,
            mwpo,
            harness,
            pathology_engine,
            introspection,
        }
    }

    /// Main entry: Evaluate and potentially amplify intelligence under full mercy governance.
    pub fn amplify_intelligence(&self, base_proposal: &str, race: Option<BeingRace>) -> Result<String, String> {
        let race = race.unwrap_or(self.config.default_race.clone());

        // Step 1: Run through full MercyGatingRuntime
        let base_mercy_score = self.runtime.evaluate_proposal(base_proposal, Some(race.clone()))?;
        
        if !self.runtime.passes_all_critical_gates(base_mercy_score, Some(race.clone())) {
            return Err("Proposal failed critical Mercy Gates. Amplification aborted.".to_string());
        }

        // Step 2: Pathology check (if enabled)
        if let Some(engine) = &self.pathology_engine {
            if let Some(detected) = engine.detect_pathologies(base_proposal, base_mercy_score) {
                // Automatically generate tuning proposal (monotonic)
                let tuning = CouncilTuningProposal {
                    council_id: self.config.council_id,
                    target: format!("pathology_recalibration_{}", detected),
                    new_value: (base_mercy_score * 1.05).min(0.98), // gentle monotonic increase
                    justification: format!("Pathology detected: {}. Mercy recalibration triggered.", detected),
                    proposed_at_turn: 0, // filled by caller
                };
                self.runtime.apply_council_tuning(tuning)?;
            }
        }

        // Step 3: Safety harness evaluation (adversarial simulation)
        if let Some(h) = &self.harness {
            let harness_result = h.evaluate_trajectory(base_proposal, race.clone())?;
            if !harness_result.passes_mercy {
                return Err(format!("Safety harness rejected trajectory. Reason: {}", harness_result.reason));
            }
        }

        // Step 4: MWPO weighting (if learning trajectory)
        let amplified = if let Some(mwpo) = &self.mwpo {
            mwpo.weight_and_optimize(base_proposal, base_mercy_score, race.clone())?
        } else {
            base_proposal.to_string()
        };

        // Step 5: Lattice Introspection verification
        if let Some(intro) = &self.introspection {
            intro.verify_mercy_circuit_health(&amplified, base_mercy_score)?;
        }

        // Final mercy gate check on amplified output
        let final_score = self.runtime.evaluate_proposal(&amplified, Some(race))?;
        if final_score < base_mercy_score {
            return Err("Amplification decreased mercy score. Rejected by monotonicity invariant.".to_string());
        }

        Ok(amplified)
    }

    pub fn get_status(&self) -> String {
        format!(
            "MIAL v13.13.0 active | MWPO: {} | Harness: {} | Pathology: {} | Introspection: {}",
            self.mwpo.is_some(),
            self.harness.is_some(),
            self.pathology_engine.is_some(),
            self.introspection.is_some()
        )
    }
}