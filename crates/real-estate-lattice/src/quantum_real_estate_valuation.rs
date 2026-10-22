//! Quantum Real Estate Valuation Engine
//! Derived from RREL documentation (April 29, 2026)
//!
//! RECO AI guidance: price and risk advice is the registrant's responsibility.
//! This module must not emit a price without a cited source and registrant sign-off.
//! Source: https://www.reco.on.ca/agents-and-brokerages/reco-bulletins/reco-guidance-use-of-artificial-intelligence/
//!
//! RREL-UNGATE-1: ungated — ra-thor-mercy and quantum-swarm imports replaced with
//! local std-only logic. analyse_property Draft behaviour preserved (#730).

use crate::RREL_VERSION;
use crate::rrel_compliance::{AiOutput, Provenance};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

// ---------------------------------------------------------------------------
// ValuationResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValuationResult {
    pub property_id: String,
    /// The price submitted by the caller as a starting reference (e.g. listing price).
    /// This is NOT an AI-generated valuation — RREL does not produce prices.
    pub reference_price: Option<f64>,
    /// Cited comparables provided by the registrant.
    pub cited_comparables: Vec<String>,
    pub regulatory_risk_score: f64,
    pub timestamp: DateTime<Utc>,
    /// Human-readable note. Always a draft until HumanVerified.
    pub draft_note: String,
}

// ---------------------------------------------------------------------------
// QuantumRealEstateValuation
// ---------------------------------------------------------------------------

/// Regulatory-risk analyser. Does not produce or adjust prices.
/// All outputs are Draft until a registered agent calls `.verify(agent_id)`.
pub struct QuantumRealEstateValuation;

impl QuantumRealEstateValuation {
    pub fn new() -> Self {
        Self
    }

    /// Analyse a property for regulatory risk.
    ///
    /// Does NOT produce or adjust a price. Price advice must come from a registered
    /// agent citing verifiable comparables, not from AI multipliers.
    ///
    /// Returns an `AiOutput<ValuationResult>` always in `Draft` status.
    /// The registrant must call `.verify(agent_id)` before sharing the output.
    pub async fn analyse_property(
        &self,
        property_id: &str,
        description: &str,
        cited_comparables: Vec<String>,
    ) -> AiOutput<ValuationResult> {
        let regulatory_risk = self.calculate_regulatory_risk(description);

        let needs_verification = vec![
            "Regulatory risk score must be reviewed by a registered agent.".to_string(),
            "Any price guidance must be sourced from verifiable comparables, not this output.".to_string(),
            "Cross-check all AI-generated descriptions against verifiable property documents \
             and physical facts before publishing.".to_string(),
        ];

        let mut sources: Vec<Provenance> = cited_comparables
            .iter()
            .map(|c| Provenance {
                label: c.clone(),
                reference: None,
                notes: Some("Registrant-supplied comparable".to_string()),
            })
            .collect();

        if sources.is_empty() {
            sources.push(Provenance {
                label: "No comparables provided".to_string(),
                reference: None,
                notes: Some(
                    "A registrant must supply verified comparables before any \
                     price guidance can be given."
                        .to_string(),
                ),
            });
        }

        let result = ValuationResult {
            property_id: property_id.to_string(),
            reference_price: None, // price requires a cited source and registrant sign-off
            cited_comparables: cited_comparables.clone(),
            regulatory_risk_score: regulatory_risk,
            timestamp: Utc::now(),
            draft_note: format!(
                "DRAFT — regulatory risk score {:.2}. No price has been generated. \
                 Price and investment advice is the registrant's responsibility, \
                 not the AI's. This output requires human verification before use. \
                 (RREL v{})",
                regulatory_risk, RREL_VERSION
            ),
        };

        AiOutput::draft(result, needs_verification, sources)
    }

    fn calculate_regulatory_risk(&self, description: &str) -> f64 {
        let mut risk: f64 = 0.25;

        if description.to_lowercase().contains("trust account") { risk += 0.20; }
        if description.to_lowercase().contains("conflict") { risk += 0.18; }
        if description.to_lowercase().contains("eviction") { risk += 0.15; }
        if description.to_lowercase().contains("advertising") { risk += 0.12; }

        risk.min(0.85)
    }
}

impl Default for QuantumRealEstateValuation {
    fn default() -> Self {
        Self::new()
    }
}
