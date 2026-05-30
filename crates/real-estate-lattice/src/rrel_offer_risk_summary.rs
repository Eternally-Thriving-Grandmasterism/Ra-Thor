//! Small helper: RrelOfferRiskSummary
//! Aggregates key signals from v14.3 modules into a single mercy-aware view.

use crate::OntarioOfferFlowReport;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfferRiskSummary {
    pub deal_type: String,
    pub recommended_form: String,
    pub offer_valid: bool,
    pub has_status_risk: bool,
    pub has_developer_risk: bool,
    pub escalation_recommended: bool,
    pub mercy_score: f64,
    pub summary: String,
}

impl OfferRiskSummary {
    pub fn from_flow_report(report: &OntarioOfferFlowReport) -> Self {
        let has_status_risk = report.status_certificate_risk.is_some();
        let has_developer_risk = report.developer_risk.is_some();

        let summary = if report.offer_valid {
            if has_developer_risk || has_status_risk {
                "Valid offer with noted risks — review recommended".to_string()
            } else {
                "Clean valid offer — proceed with confidence".to_string()
            }
        } else {
            "Offer requires attention before proceeding".to_string()
        };

        Self {
            deal_type: report.deal_type.clone(),
            recommended_form: report.recommended_form.clone(),
            offer_valid: report.offer_valid,
            has_status_risk,
            has_developer_risk,
            escalation_recommended: report.multi_offer_escalation_triggered,
            mercy_score: report.overall_mercy,
            summary,
        }
    }
}
