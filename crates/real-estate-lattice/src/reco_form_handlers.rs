//! Design-only (STUB-AUDIT-1 WORKSPACE-KEY-SHELLS-1 council note): not compiled; inherits workspace key powrush that the root does not define.
//! RECO/TRESA form drafting modules (unverified) — RREL v3.1.
//! Handles Form 801 and additional common RECO forms with full audit trail.
//! Privacy-first, TOLC 8 checks planned (design-only, unverified).
//!
//! RECO AI guidance: forms are AI-drafted only; a registered agent must review
//! and sign off before submission. Compliance is NOT verified by this module.
//! Source: https://www.reco.on.ca/agents-and-brokerages/reco-bulletins/reco-guidance-use-of-artificial-intelligence/

use crate::compliance_helpers::ComplianceResult;
use crate::rrel_compliance::{AiOutput, AuditLog, ConsentRecord, Provenance};
use mercy::traits::{MercyAligned, TOLC8Gate};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RecoForm {
    pub form_id: String,
    pub form_type: RecoFormType,
    pub data: HashMap<String, String>,
    pub compliance: ComplianceResult,
    /// Seal is never set by AI. A registered agent sets this after review.
    pub tolc8_seal: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecoFormType {
    Form801,
    ListingAgreement,
    BuyerRepresentation,
    OfferToPurchase,
    Custom(String),
}

pub struct RecoFormHandlers {
    pub audit_log: AuditLog,
}

impl RecoFormHandlers {
    pub fn new() -> Self {
        Self {
            audit_log: AuditLog::new(),
        }
    }

    /// Process a RECO form with consent and verification gates.
    ///
    /// Requires written client consent before any client data is processed.
    /// Always returns a Draft output — the registrant must verify before submission.
    pub fn process_reco_form(
        &mut self,
        form_type: RecoFormType,
        data: HashMap<String, String>,
        consent: &ConsentRecord,
    ) -> Result<AiOutput<RecoForm>, String> {
        // Consent gate — written client consent required (RECO AI guidance)
        consent.gate()?;

        if data.is_empty() {
            return Err("Form data cannot be empty".to_string());
        }

        let compliance = self.validate_reco_compliance(&form_type, &data);

        let form = RecoForm {
            form_id: format!("RECO-{}-{:?}", chrono::Utc::now().timestamp(), form_type),
            form_type,
            data,
            compliance,
            tolc8_seal: false, // never set by AI; agent sets this after independent review
        };

        let needs_verification = vec![
            "RECO/TRESA compliance has NOT been verified by this module. A registered \
             agent must independently review all form fields before submission.".to_string(),
            "Cross-check all auto-populated fields against source documents.".to_string(),
            "Confirm all disclosure obligations are met per TRESA.".to_string(),
        ];

        let sources = vec![Provenance {
            label: "Agent-supplied form data".to_string(),
            reference: None,
            notes: Some("No AI-generated client data included.".to_string()),
        }];

        Ok(AiOutput::draft(form, needs_verification, sources))
    }

    fn validate_reco_compliance(
        &self,
        _form_type: &RecoFormType,
        _data: &HashMap<String, String>,
    ) -> ComplianceResult {
        // Not implemented — real RECO/TRESA rule checks require a licensed
        // compliance engine. Returning unverified so callers are not misled.
        ComplianceResult {
            passed: false,
            issues: vec![
                "Compliance not implemented — human review required before submission. \
                 This module does not verify RECO/TRESA rules."
                    .to_string(),
            ],
            score: 0.0,
        }
    }
}

impl MercyAligned for RecoFormHandlers {
    fn check_mercy_gates(&self) -> Vec<TOLC8Gate> {
        vec![
            TOLC8Gate::Genesis,
            TOLC8Gate::Truth,
            TOLC8Gate::Evolution,
            TOLC8Gate::Sovereignty,
            TOLC8Gate::Infinite,
        ]
    }
}
