/*!
 * RREL Reference Generator (v1.1.0)
 * Professional reference / summary generator for real estate forms and compliance packages.
 * Now with actual Markdown output support.
 * Privacy-first, example-only, sovereign, mercy-gated.
 */

use chrono::{DateTime, Utc};
use crate::rrel_form801_preset::{Form801Preset, SubmissionTrack};
use crate::rrel_offer_package::OfferPackage;
use crate::rrel_compliance_helpers::{RecordRetentionMetadata, RetentionCategory};

/// Generates a clean, professional plain-text reference summary.
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
        output.push_str(ret.generate_retention_note());
    }

    output.push_str("\n=== END REFERENCE ===\n");
    output.push_str("\nNote: This is an example-only reference. All data is fictional. For professional use only.\n");

    output
}

/// Generates a professional Markdown reference document.
/// Ready for rendering, saving as .md, or conversion to DOCX/PDF.
pub fn generate_markdown_reference(
    preset: &Form801Preset,
    offer_package: Option<&OfferPackage>,
    retention: Option<&RecordRetentionMetadata>,
    compliance_notes: &[String],
) -> String {
    let mut md = String::new();

    md.push_str("# RREL Professional Reference Summary\n\n");
    md.push_str(&format!("**Generated:** {}\n\n", Utc::now().format("%Y-%m-%d %H:%M UTC")));

    md.push_str("## Form 801 Preset\n\n");
    md.push_str(&format!("- **Track:** `{:?}`\n", preset.track));
    md.push_str(&format!("- **Property Address:** {}\n", preset.property_address));
    md.push_str(&format!("- **Buyer(s):** {}\n", preset.buyer_names.join(", ")));
    md.push_str(&format!("- **Irrevocable Until:** {}\n\n", preset.irrevocable_until));

    if let Some(pkg) = offer_package {
        md.push_str("## Offer Package\n\n");
        md.push_str(&format!("- **Cross-Validation Passed:** {}\n", pkg.cross_validation_passed));
        if let Some(report) = &pkg.validation_report {
            md.push_str(&format!("- **Validation Report:** {}\n", report));
        }
        md.push_str("\n");
    }

    if let Some(ret) = retention {
        md.push_str("## Record Retention\n\n");
        md.push_str(&format!("- **Category:** `{:?}`\n", ret.category));
        md.push_str(&format!("- **Retention Period:** {} years\n", ret.retention_years));
        md.push_str(&format!("- **Note:** {}\n\n", ret.generate_retention_note().trim()));
    }

    if !compliance_notes.is_empty() {
        md.push_str("## Compliance Notes\n\n");
        for note in compliance_notes {
            md.push_str(&format!("- {}\n", note));
        }
        md.push_str("\n");
    }

    md.push_str("---\n\n");
    md.push_str("**Note:** This is an example-only reference document. All data is fictional. ");
    md.push_str("Intended for professional real estate use in alignment with RECO/TRESA and Ra-Thor Eternal principles.\n");

    md
}

/// Future extension point for PDF generation (placeholder).
pub fn generate_pdf_ready_content(summary: &str) -> String {
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
    }

    #[test]
    fn test_markdown_reference_generation() {
        let preset = Form801Preset::new_standard(
            "456 Oak Lane, Ottawa, ON".to_string(),
            vec!["Jane & Family Buyer".to_string()],
            "2026-06-20 17:00".to_string(),
        );
        let md = generate_markdown_reference(&preset, None, None, &vec![]);
        assert!(md.contains("# RREL Professional Reference Summary"));
        assert!(md.contains("**Track:**"));
    }
}
