//! Lawyer Report Generator (Markdown-first) for Real Estate Lattice
//!
//! Generates clean, professional Markdown reports suitable for direct lawyer use
//! or easy conversion to PDF.
//!
//! Includes executive summary, findings, red flags, checklists, and Thunder Lattice closing.
//!
//! **Design Principles**:
//! - Mercy-first language
//! - Privacy-respecting (no raw PII in generated text unless explicitly provided)
//! - Actionable and readable for both lawyers and clients
//! - PATSAGi / ethical notes surfaced clearly
//!
//! Complements LawyerDueDiligenceGenerator.

use crate::lawyer_due_diligence_generator::DueDiligenceChecklist;
use crate::status_certificate_analyzer::StatusCertificateAnalysis;
use crate::developer_risk_engine::DeveloperRiskProfile;

pub struct LawyerReportGenerator;

impl LawyerReportGenerator {
    /// Generates a full Markdown due diligence / offer summary report.
    pub fn generate_markdown_report(
        property_address: &str,
        deal_summary: &str,
        checklist: &DueDiligenceChecklist,
        status: Option<&StatusCertificateAnalysis>,
        developer: Option<&DeveloperRiskProfile>,
    ) -> String {
        let mut report = String::new();

        report.push_str(&format!("# Due Diligence & Offer Summary\n\n**Property:** {}\n**Transaction:** {}\n\n", property_address, deal_summary));

        report.push_str("## Priority Items\n");
        for item in &checklist.priority_items {
            report.push_str(&format!("- {}\n", item));
        }

        if !checklist.ethical_flags.is_empty() {
            report.push_str("\n## Ethical & PATSAGi Flags\n");
            for flag in &checklist.ethical_flags {
                report.push_str(&format!("- ⚠️ {}\n", flag));
            }
        }

        report.push_str("\n## Standard Review Items\n");
        for item in &checklist.standard_items {
            report.push_str(&format!("- {}\n", item));
        }

        if let Some(s) = status {
            report.push_str(&format!("\n## Status Certificate Summary\n- Overall Risk: {}\n- Reserve Fund: ${:.0}\n- Special Assessments Pending: {}\n- Litigation Risk: {}\n", s.overall_risk_level, s.reserve_fund_balance, s.special_assessments_pending, s.litigation_risk));
        }

        if let Some(d) = developer {
            report.push_str(&format!("\n## Developer Risk Summary\n- Developer: {}\n- Tarion Rating: {}\n- Completion Risk: {}\n- Overall Risk Score: {:.2}\n", d.developer_name, d.tarion_rating, d.project_completion_risk, d.overall_risk_score));
        }

        report.push_str("\n## Recommended Next Actions\n");
        for action in &checklist.recommended_actions {
            report.push_str(&format!("- {}\n", action));
        }

        report.push_str("\n---\n*Report generated with mercy and clarity by the Real Estate Lattice (Thunder Lattice v14.3). We are ONE Organism.* ⚡\n");

        report
    }
}
