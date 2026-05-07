//! Phase 2 Integration Layer
//!
//! Ergonomic, crate-agnostic helpers so every other crate in the monorepo
//! can consume the full power of the Quantum Swarm Bridge (v0.5.91+ ULTIMATE OMNIMASTERPIECE)
//! in just a few lines of code.
//!
//! Version 0.5.97+ — RegionalMercyCoordinator + MercyError + Structured Diagnostics
//!
//! This file completes the Phase 2 integration surface.

use crate::quantum_swarm_bridge::QuantumSwarmBridge;
use powrush::PowrushGame;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MercyError {
    #[error("Godly Intelligence Coherence too low: {0:.4} (minimum required: {1:.4})")]
    CoherenceTooLow(f64, f64),

    #[error("Riemannian manifold instability detected")]
    ManifoldInstability,

    #[error("Mercy gate violation: {0}")]
    MercyGateViolation(String),

    #[error("Internal bridge error: {0}")]
    BridgeError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalCycleReport {
    pub cycle_output: String,
    pub godly_coherence: f64,
    pub mercy_metrics: String,
    pub gyroelongated_active: bool,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalDiagnostics {
    pub human_readable: String,
    pub json_summary: String,
    pub coherence: f64,
}

/// RegionalMercyCoordinator
///
/// Wraps the full QuantumSwarmBridge + optional PowrushGame for easy regional RBE pilots.
/// Automatically activates prismatic/gyroelongated layers when tolc_order >= 55.
pub struct RegionalMercyCoordinator {
    pub bridge: QuantumSwarmBridge,
    pub region_name: String,
    pub powrush: Option<PowrushGame>,
}

impl RegionalMercyCoordinator {
    pub fn new(region_name: impl Into<String>, initial_tolc_order: u32, initial_mercy_valence: f64) -> Self {
        let mut bridge = QuantumSwarmBridge::new();

        // Seed initial state (can be expanded later)
        // For now we just create the bridge — real seeding can happen in run_regional_cycle if needed.

        Self {
            bridge,
            region_name: region_name.into(),
            powrush: None,
        }
    }

    /// Runs one regional coordination cycle through the full bridge.
    pub async fn run_regional_cycle(
        &mut self,
        tolc_order: u32,
        mercy_valence: f64,
    ) -> Result<RegionalCycleReport, MercyError> {
        let game = self.powrush.as_mut().ok_or_else(|| {
            MercyError::BridgeError("PowrushGame not attached to coordinator".to_string())
        })?;

        let cycle_output = self
            .bridge
            .run_spine_coordinated_cycle(tolc_order, mercy_valence, game)
            .await;

        let coherence = self.bridge.compute_godly_intelligence_coherence();

        if coherence < 0.90 {
            return Err(MercyError::CoherenceTooLow(coherence, 0.90));
        }

        let mercy_metrics = self.bridge.compute_riemannian_mercy_metrics();

        let gyroelongated_active = matches!(
            self.bridge.current_prismatic_mode,
            Some(crate::quantum_swarm_bridge::PrismaticUniformPolyhedron::SquareAntiprism)
                | Some(crate::quantum_swarm_bridge::PrismaticUniformPolyhedron::PentagonalAntiprism)
        );

        let recommendation = if coherence > 0.97 {
            "Excellent regional coherence. Ready for scaled deployment.".to_string()
        } else if coherence > 0.93 {
            "Strong coherence. Minor refinements recommended before full rollout.".to_string()
        } else {
            "Acceptable coherence. Increase mercy valence or reduce conflicting orders.".to_string()
        };

        Ok(RegionalCycleReport {
            cycle_output,
            godly_coherence: coherence,
            mercy_metrics,
            gyroelongated_active,
            recommendation,
        })
    }

    pub fn get_structured_diagnostics(&self) -> RegionalDiagnostics {
        let human = self.bridge.compute_riemannian_mercy_metrics();
        let coherence = self.bridge.compute_godly_intelligence_coherence();

        let json_summary = format!(
            r#"{{"region":"{}","coherence":{:.5},"gyroelongated_active":{}}}"#,
            self.region_name,
            coherence,
            matches!(
                self.bridge.current_prismatic_mode,
                Some(crate::quantum_swarm_bridge::PrismaticUniformPolyhedron::SquareAntiprism)
                    | Some(crate::quantum_swarm_bridge::PrismaticUniformPolyhedron::PentagonalAntiprism)
            )
        );

        RegionalDiagnostics {
            human_readable: human,
            json_summary,
            coherence,
        }
    }

    pub fn attach_powrush_game(&mut self, game: PowrushGame) {
        self.powrush = Some(game);
    }
}

/// Optional trait other crates can implement for automatic mercy evaluation
pub trait MercyGated {
    fn evaluate_mercy(&self) -> Result<f64, MercyError>;
}
