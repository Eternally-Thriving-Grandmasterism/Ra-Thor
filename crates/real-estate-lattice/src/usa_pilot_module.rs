//! UsaPilotModule — Central Orchestrator for USA Real Estate Offer Processing
//!
//! This module provides the main entry point for processing USA real estate offers
//! in a mercy-gated, cached, and extensible way.
//!
//! It combines:
//! - Regulatory checks via `UsaRegulatoryEngine`
//! - External data enrichment via `AttomDataProvider` + `AttomCache`
//! - Geometric / Sacred Harmony assessment via `GeometricHarmonyAdvisor` (v14.4)
//!
//! Part of PR #192 — v14.3+ Execution Stabilization + ONE Organism Geometric Integration.

use crate::geometric_harmony_advisor::GeometricHarmonyAdvisor;
use crate::usa_attom_data_provider::AttomDataProvider;
use crate::usa_regulatory_engine::UsaRegulatoryEngine;
use crate::usa_state_adapters::{UsaStateAdapters, UsState};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsaPilotReport {
    // Placeholder for future aggregated pilot reporting
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsaOfferFlowReport {
    pub state: String,
    pub passed_regulatory: bool,
    pub mercy_valence: f64,
    pub quantum_consensus: f64,
    pub federal_issues: Vec<String>,
    pub state_issues: Vec<String>,
    pub summary: String,

    /// Enriched data from external provider (e.g. ATTOM via cache)
    pub external_property_profile: Option<crate::usa_attom_cache::PropertyProfile>,
    pub external_risk_signals: Option<crate::usa_attom_cache::RiskSignals>,
}

pub struct UsaPilotModule {
    state_adapters: UsaStateAdapters,
    regulatory_engine: UsaRegulatoryEngine,
    data_provider: AttomDataProvider,

    // === NEW: Geometric Harmony Advisor (v14.4) ===
    geometric_advisor: GeometricHarmonyAdvisor,
}

impl UsaPilotModule {
    pub fn new(
        mercy_engine: MercyEngine,
        quantum_swarm: QuantumSwarmOrchestrator,
        world_governance: WorldGovernanceEngine,
    ) -> Self {
        Self {
            state_adapters: UsaStateAdapters::new(
                mercy_engine.clone(),
                quantum_swarm.clone(),
                world_governance.clone(),
            ),
            regulatory_engine: UsaRegulatoryEngine::new(
                mercy_engine,
                quantum_swarm,
                world_governance,
            ),
            data_provider: AttomDataProvider::new(),

            // === NEW ===
            geometric_advisor: GeometricHarmonyAdvisor::new(),
        }
    }

    /// Process a USA offer flow with optional external data enrichment.
    ///
    /// When `property_identifier` is provided, the module will attempt to
    /// enrich the report using the configured data provider (currently stubbed
    /// with caching). This enables richer due diligence and risk assessment.
    pub async fn process_usa_offer_flow(
        &mut self,
        state: UsState,
        transaction_details: &str,
        price: f64,
        game: &mut PowrushGame,
        property_identifier: Option<&str>,
    ) -> Result<UsaOfferFlowReport, crate::RrelError> {
        info!("🇺🇸 Processing USA offer flow for {:?}", state);

        let regulatory_result = self.regulatory_engine
            .check_usa_transaction(&format!("{:?}", state), transaction_details, price, game)
            .await
            .map_err(|e| crate::RrelError::Other(format!("Regulatory check failed: {}", e)))?;

        // Optional enrichment via cached data provider
        let mut external_profile = None;
        let mut external_risk = None;

        if let Some(identifier) = property_identifier {
            if let Ok(profile) = self.data_provider.get_property_profile(state, identifier).await {
                external_profile = Some(profile);
            }
            if let Ok(signals) = self.data_provider.get_risk_signals(state, identifier).await {
                external_risk = Some(signals);
            }
        }

        let summary = if regulatory_result.passed {
            format!("USA offer cleared regulatory checks in {:?}", state)
        } else {
            format!("USA offer has regulatory issues in {:?}", state)
        };

        Ok(UsaOfferFlowReport {
            state: format!("{:?}", state),
            passed_regulatory: regulatory_result.passed,
            mercy_valence: regulatory_result.mercy_valence,
            quantum_consensus: regulatory_result.quantum_consensus,
            federal_issues: regulatory_result.federal_issues,
            state_issues: regulatory_result.state_issues,
            summary,
            external_property_profile: external_profile,
            external_risk_signals: external_risk,
        })
    }

    /// NEW (v14.4): Assess geometric / sacred harmony for a property or site.
    ///
    /// This brings ONE Organism geometric intelligence (PolyhedralHarmonicEngine + U57 readiness)
    /// directly into the USA Pilot workflow. Useful for spatial harmony evaluation,
    /// layout recommendations, and future mercy-aligned valuation enhancements.
    pub fn assess_geometric_harmony(
        &self,
        tolc_order: u32,
        base_coherence: f64,
    ) -> crate::geometric_harmony_advisor::GeometricHarmonyAssessment {
        self.geometric_advisor.assess_property_harmony(tolc_order, base_coherence)
    }

    /// NEW (v14.4): Enriched version of offer flow that also returns geometric harmony assessment.
    /// Can be used when deeper due diligence including sacred geometry is desired.
    pub async fn process_usa_offer_flow_with_geometric_harmony(
        &mut self,
        state: UsState,
        transaction_details: &str,
        price: f64,
        game: &mut PowrushGame,
        property_identifier: Option<&str>,
        tolc_order: u32,
    ) -> Result<(UsaOfferFlowReport, Option<crate::geometric_harmony_advisor::GeometricHarmonyAssessment>), crate::RrelError> {
        
        let report = self.process_usa_offer_flow(
            state,
            transaction_details,
            price,
            game,
            property_identifier,
        ).await?;

        let geometric_assessment = if tolc_order >= 8 {
            Some(self.assess_geometric_harmony(tolc_order, 0.91))
        } else {
            None
        };

        Ok((report, geometric_assessment))
    }
}
