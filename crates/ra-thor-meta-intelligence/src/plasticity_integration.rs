// crates/ra-thor-meta-intelligence/src/plasticity_integration.rs

use crate::self_improvement_engine::{ImprovementProposal, ImprovementType};
use plasticity_engine_v2::PlasticityEngineV2;
use mercy_merlin_engine::MercyMerlinEngine;

/// Bridges SelfImprovementEngine with PlasticityEngineV2.
/// Creates a feedback loop where plasticity can influence proposal priority
/// and successful improvements can strengthen long-term epigenetic traits.
pub struct PlasticityIntegration {
    plasticity_engine: PlasticityEngineV2,
    mercy_engine: MercyMerlinEngine,
}

impl PlasticityIntegration {
    pub fn new(plasticity_engine: PlasticityEngineV2, mercy_engine: MercyMerlinEngine) -> Self {
        Self { plasticity_engine, mercy_engine }
    }

    /// Modulates proposal priority using plasticity rules (mercy-gated).
    pub async fn modulate_proposal_priority(
        &self,
        proposal: &mut ImprovementProposal,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let valence = self.mercy_engine.get_current_valence().await?;
        if valence < 0.92 {
            return Ok(());
        }

        if proposal.mercy_alignment >= 9 {
            proposal.priority_score *= 1.15;
        }
        if proposal.improvement_type == ImprovementType::MercyIntegration {
            proposal.priority_score *= 1.10;
        }
        Ok(())
    }

    /// Records a successful improvement as an epigenetic update (mercy-gated).
    pub async fn record_successful_improvement(
        &mut self,
        proposal: &ImprovementProposal,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let valence = self.mercy_engine.get_current_valence().await?;
        if valence < 0.92 {
            return Ok(());
        }

        // Placeholder for real epigenetic update call
        // self.plasticity_engine.apply_epigenetic_update(...).await?;
        tracing::info!("Epigenetic update triggered from successful improvement: {}", proposal.title);
        Ok(())
    }
}