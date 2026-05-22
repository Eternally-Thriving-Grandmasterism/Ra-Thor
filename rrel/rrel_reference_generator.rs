/*!
 * RREL Reference Generator Skeleton (v1.0.0)
 * Professional reference / summary generator for real estate forms and compliance packages.
 * Privacy-first, example-only, sovereign, mercy-gated.
 *
 * Future: Markdown, HTML, and PDF-ready output templates.
 */

use chrono::{DateTime, Utc};
use crate::rrel_form801_preset::{Form801Preset, SubmissionTrack};
use crate::rrel_offer_package::OfferPackage;
use crate::rrel_compliance_helpers::{RecordRetentionMetadata, RetentionCategory};

/// Generates a clean, professional reference summary for a Form 801 + Offer Package.
pub fn generate_form801_reference_summary(
    preset: &Form801Preset,
    offer_package: Option<&OfferPackage>,
    retention: Option<&RecordRetentionMetadata>,
) -> String {
    let mut output = String::new();

    output.push_str("=== RREL PROFESSIONAL REFERENCE SUMMARY ===\n");
    output.push_str(&format!("Generated: {}\n\n", Utc::now().format("%Y-%m-%d %H:%M UTC")));

    output.push_str("--- FORM 801 PRESET ---\n");
    output.push_str(&format!("Track: {:?}\n", preset.track));
    output.push_str(&format!("Property Address: {}\n", preset.property_address));
    output.push_str(&format!("Buyer(s): {}\n", preset.buyer_names.join(", ")));
    output.push_str(&format!("Irrevocable Until: {}\n", preset.irrevocable_until));

    if let Some(pkg) = offer_package {
        output.push_str("\n--- OFFER PACKAGE ---\n");
        output.push_str(&format!("Cross-Validation Passed: {}\n", pkg.cross_validation_passed));
        if let Some(report) = &pkg.validation_report {
            output.push_str(&format!("Validation Report: {}\n", report));
        }
    }

    if let Some(ret) = retention {
        output.push_str("\n--- RECORD RETENTION ---\n");
        output.push_str(&format!("Category: {:?}\n", ret.category));
        output.push_str(&format!("Retention Period: {} years\n", ret.retention_years));
        output.push_str(&ret.generate_retention_note());
    }

    output.push_str("\n=== END REFERENCE ===\n");
    output.push_str("\nNote: This is an example-only reference. All data is fictional. For professional use only.\n");

    output
}

/// Future extension point for PDF generation (placeholder).
pub fn generate_pdf_ready_content(summary: &str) -> String {
    // In future: integrate with printpdf, lopdf, or external service
    format!("PDF-READY CONTENT (skeleton):\n{}\n", summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rrel_form801_preset::Form801Preset;

    #[test]
    fn test_reference_generator_basic() {
        let preset = Form801Preset::new_standard(
            "123 Example St, Toronto, ON".to_string(),
            vec!["John Buyer".to_string()],
            "2026-06-15 23:59".to_string(),
        );
        let summary = generate_form801_reference_summary(&preset, None, None);
        assert!(summary.contains("FORM 801 PRESET"));
        assert!(summary.contains("123 Example St"));
    }
}
